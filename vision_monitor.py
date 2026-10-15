from __future__ import annotations

import math
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from config import Config, MODEL_URL
from models import CalibratedMeasurement, GazeMeasurement, RiskEvent, VisionFrameResult


def ensure_model(config: Config) -> None:
    if config.model_path.exists():
        return
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    print("MediaPipe 얼굴 모델 다운로드 중...")
    urllib.request.urlretrieve(MODEL_URL, config.model_path)
    print("모델 다운로드 완료")


class MeasurementSmoother:
    def __init__(self, alpha: float):
        self.alpha = max(0.0, min(1.0, alpha))
        self.previous: GazeMeasurement | None = None

    def update(self, current: GazeMeasurement) -> GazeMeasurement:
        if self.previous is None:
            self.previous = current
            return current
        a, p = self.alpha, self.previous
        result = GazeMeasurement(
            a * current.raw_yaw + (1.0 - a) * p.raw_yaw,
            a * current.raw_pitch + (1.0 - a) * p.raw_pitch,
            a * current.gaze_x + (1.0 - a) * p.gaze_x,
            a * current.gaze_y + (1.0 - a) * p.gaze_y,
            a * current.ear + (1.0 - a) * p.ear,
            current.face_quality, current.bbox, current.iris_points,
        )
        self.previous = result
        return result

    def reset(self) -> None:
        self.previous = None


class Calibrator:
    def __init__(self, config: Config):
        self.config = config
        self.started_at: float | None = None
        self.samples: list[tuple[float, float, float, float, float]] = []
        self.verify_started_at: float | None = None
        self.verify_total = 0
        self.verify_passed = 0
        self.ready = False
        self.baseline: dict[str, float] = {}
        self.phase = "COLLECTING"
        self.status = "LOOK FORWARD AND HOLD STILL"
        self.attempt = 1

    def reset(self, reason: str = "MANUAL RESET") -> None:
        self.started_at = None
        self.samples.clear()
        self.verify_started_at = None
        self.verify_total = 0
        self.verify_passed = 0
        self.ready = False
        self.baseline = {}
        self.phase = "COLLECTING"
        self.status = f"RETRY: {reason} - LOOK FORWARD"
        self.attempt += 1

    @staticmethod
    def _median_and_mad(samples) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(samples, dtype=np.float64)
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        return median, mad

    def _candidate_error(self, median: np.ndarray, mad: np.ndarray) -> str:
        if abs(float(median[0])) > self.config.calibration_max_abs_yaw:
            return "FACE NOT CENTERED HORIZONTALLY"
        if abs(float(median[1])) > self.config.calibration_max_abs_pitch:
            return "CAMERA HEIGHT OR HEAD TILT"
        if float(mad[0]) > self.config.calibration_max_yaw_mad:
            return "HEAD MOVED LEFT OR RIGHT"
        if float(mad[1]) > self.config.calibration_max_pitch_mad:
            return "HEAD MOVED UP OR DOWN"
        if max(float(mad[2]), float(mad[3])) > self.config.calibration_max_gaze_mad:
            return "EYES MOVED DURING CALIBRATION"
        return ""

    def _start_verification(self, median: np.ndarray, now: float) -> None:
        self.baseline = {
            "yaw": float(median[0]), "pitch": float(median[1]),
            "gaze_x": float(median[2]), "gaze_y": float(median[3]),
            "ear": float(median[4]),
        }
        self.phase = "VERIFYING"
        self.verify_started_at = now
        self.verify_total = 0
        self.verify_passed = 0
        self.status = "VERIFYING CENTER - KEEP STILL"

    def _verification_passes(self, measurement: GazeMeasurement) -> bool:
        return (
            abs(measurement.raw_yaw - self.baseline["yaw"])
            <= self.config.calibration_verify_yaw
            and abs(measurement.raw_pitch - self.baseline["pitch"])
            <= self.config.calibration_verify_pitch
            and abs(measurement.gaze_x - self.baseline["gaze_x"])
            <= self.config.calibration_verify_gaze
            and abs(measurement.gaze_y - self.baseline["gaze_y"])
            <= self.config.calibration_verify_gaze
        )

    def add(self, measurement: GazeMeasurement, now: float) -> str | None:
        if self.ready:
            return None
        if measurement.face_quality < self.config.min_face_quality:
            self.status = "FACE TOO SMALL - MOVE CLOSER"
            return None
        if measurement.ear <= 0.05:
            self.status = "OPEN YOUR EYES"
            return None

        if self.phase == "VERIFYING":
            self.verify_total += 1
            if self._verification_passes(measurement):
                self.verify_passed += 1
            elapsed = now - float(self.verify_started_at)
            enough = (
                elapsed >= self.config.calibration_verify_sec
                and self.verify_total >= self.config.calibration_verify_min_samples
            )
            if not enough:
                return None
            pass_ratio = self.verify_passed / max(1, self.verify_total)
            if pass_ratio >= self.config.calibration_min_pass_ratio:
                self.ready = True
                self.phase = "READY"
                self.status = "CALIBRATION COMPLETE"
                return "READY"
            reason = f"VERIFY FAILED ({pass_ratio * 100:.0f}% STABLE)"
            self.reset(reason)
            return reason

        if self.started_at is None:
            self.started_at = now
        self.samples.append((
            measurement.raw_yaw, measurement.raw_pitch,
            measurement.gaze_x, measurement.gaze_y, measurement.ear,
        ))
        if not (
            now - self.started_at >= self.config.calibration_sec
            and len(self.samples) >= self.config.calibration_min_samples
        ):
            return None
        median, mad = self._median_and_mad(self.samples)
        error = self._candidate_error(median, mad)
        if error:
            self.reset(error)
            return error
        self._start_verification(median, now)
        return "VERIFYING"

    def progress(self, now: float) -> float:
        if self.ready:
            return 1.0
        if self.phase == "VERIFYING" and self.verify_started_at is not None:
            elapsed = (now - self.verify_started_at) / self.config.calibration_verify_sec
            sample_part = self.verify_total / self.config.calibration_verify_min_samples
            return 0.75 + 0.25 * max(0.0, min(1.0, min(elapsed, sample_part)))
        if self.started_at is None:
            return 0.0
        time_part = (now - self.started_at) / self.config.calibration_sec
        sample_part = len(self.samples) / self.config.calibration_min_samples
        return 0.75 * max(0.0, min(1.0, min(time_part, sample_part)))

    def apply(self, measurement: GazeMeasurement) -> CalibratedMeasurement:
        b = self.baseline
        yaw = measurement.raw_yaw - b["yaw"]
        pitch = measurement.raw_pitch - b["pitch"]
        gaze_dx = measurement.gaze_x - b["gaze_x"]
        gaze_dy = measurement.gaze_y - b["gaze_y"]
        eyes_closed = measurement.ear < b["ear"] * self.config.eye_closed_ratio

        if yaw <= -self.config.head_right_yaw_threshold:
            head = "RIGHT"
        elif yaw >= self.config.head_left_yaw_threshold:
            head = "LEFT"
        elif pitch > self.config.head_pitch_threshold:
            head = "UP"
        elif pitch < -self.config.head_pitch_threshold:
            head = "DOWN"
        else:
            head = "CENTER"

        if eyes_closed:
            gaze = "EYES_CLOSED"
        elif gaze_dx < -self.config.gaze_x_threshold:
            gaze = "LEFT"
        elif gaze_dx > self.config.gaze_x_threshold:
            gaze = "RIGHT"
        elif gaze_dy < -self.config.gaze_y_threshold:
            gaze = "UP"
        elif gaze_dy > self.config.gaze_y_threshold:
            gaze = "DOWN"
        else:
            gaze = "CENTER"
        return CalibratedMeasurement(
            yaw, pitch, gaze_dx, gaze_dy, measurement.ear, eyes_closed,
            measurement.face_quality, head, gaze,
        )


class SustainedEventTracker:
    def __init__(self):
        self.states: dict[str, dict[str, float | bool | None]] = {}

    def update(self, *, kind: str, active: bool, now: float,
               min_duration: float, repeat_sec: float, weight: float,
               value: str, confidence: float, context: str,
               release_grace_sec: float = 0.0) -> list[RiskEvent]:
        state = self.states.setdefault(
            kind, {"started": None, "last_emit": None, "emitted": False,
                   "inactive_since": None}
        )
        events: list[RiskEvent] = []
        if active:
            inactive_since = state.get("inactive_since")
            if inactive_since is not None and now - float(inactive_since) > release_grace_sec:
                if bool(state["emitted"]):
                    events.append(RiskEvent(
                        now, "vision", f"{kind}_END", 0.0, "NORMALIZED",
                        float(inactive_since) - float(state["started"]), confidence,
                        f"{kind} 상태 종료",
                    ))
                state.update(started=None, last_emit=None, emitted=False)
            state["inactive_since"] = None
            if state["started"] is None:
                state.update(started=now, last_emit=None, emitted=False)
            duration = now - float(state["started"])
            initial = duration >= min_duration and state["last_emit"] is None
            repeated = (
                state["last_emit"] is not None
                and now - float(state["last_emit"]) >= repeat_sec
            )
            if initial or repeated:
                state.update(last_emit=now, emitted=True)
                events.append(RiskEvent(
                    now, "vision", kind, weight, value, duration,
                    confidence, context,
                ))
        else:
            if state["started"] is not None and release_grace_sec > 0:
                if state.get("inactive_since") is None:
                    state["inactive_since"] = now
                if now - float(state["inactive_since"]) < release_grace_sec:
                    return events
            if state["started"] is not None and bool(state["emitted"]):
                end_at = state.get("inactive_since")
                duration = (float(end_at) if end_at is not None else now) - float(state["started"])
                events.append(RiskEvent(
                    now, "vision", f"{kind}_END", 0.0, "NORMALIZED",
                    duration, confidence, f"{kind} 상태 종료",
                ))
            state.update(started=None, last_emit=None, emitted=False, inactive_since=None)
        return events


def _point(landmarks, index: int) -> np.ndarray:
    item = landmarks[index]
    return np.asarray([item.x, item.y], dtype=np.float64)


def _eye_aspect_ratio(landmarks, indices) -> float:
    p1, p2, p3, p4, p5, p6 = [_point(landmarks, index) for index in indices]
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) /
                 (2.0 * horizontal))


def _iris_ratio(landmarks, iris_index: int, corner_a: int, corner_b: int,
                upper: int, lower: int) -> tuple[float, float]:
    iris = _point(landmarks, iris_index)
    c1, c2 = _point(landmarks, corner_a), _point(landmarks, corner_b)
    top, bottom = _point(landmarks, upper), _point(landmarks, lower)
    min_x, max_x = sorted((c1[0], c2[0]))
    min_y, max_y = sorted((top[1], bottom[1]))
    width, height = max(max_x - min_x, 1e-6), max(max_y - min_y, 1e-6)
    return float((iris[0] - min_x) / width), float((iris[1] - min_y) / height)


def _estimate_head_pose(landmarks, width: int, height: int) -> tuple[float, float] | None:
    indices = [1, 152, 33, 263, 61, 291]
    face_3d = np.asarray([
        [0.0, 0.0, 0.0], [0.0, 330.0, -65.0],
        [-225.0, -170.0, -135.0], [225.0, -170.0, -135.0],
        [-150.0, 150.0, -125.0], [150.0, 150.0, -125.0],
    ], dtype=np.float64)
    face_2d = np.asarray([
        [landmarks[index].x * width, landmarks[index].y * height]
        for index in indices
    ], dtype=np.float64)
    focal = width / (2.0 * math.tan(math.radians(60.0) / 2.0))
    camera = np.asarray([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # Initialize from a perspective solution rather than the iterative solver's
    # default seed, which can converge behind the camera with this face model.
    success, rotation_vector, translation = cv2.solvePnP(
        face_3d, face_2d, camera, np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None
    success, rotation_vector, translation = cv2.solvePnP(
        face_3d, face_2d, camera, np.zeros((4, 1), dtype=np.float64),
        rotation_vector, translation, True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success or not np.isfinite(rotation_vector).all() or not np.isfinite(translation).all():
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    depths = (rotation_matrix @ face_3d.T + translation)[2]
    if np.any(depths <= 0):
        return None
    projected, _ = cv2.projectPoints(
        face_3d, rotation_vector, translation, camera, np.zeros((4, 1)),
    )
    error = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - face_2d) ** 2, axis=1))))
    face_span = float(np.linalg.norm(np.ptp(face_2d, axis=0)))
    if not math.isfinite(error) or error > max(3.0, face_span * 0.10):
        return None
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    return float(angles[1]), float(angles[0])


def _bbox(landmarks, width: int, height: int) -> tuple[int, int, int, int]:
    xs, ys = [item.x for item in landmarks], [item.y for item in landmarks]
    return (
        max(0, int(min(xs) * width) - 10),
        max(0, int(min(ys) * height) - 10),
        min(width - 1, int(max(xs) * width) + 10),
        min(height - 1, int(max(ys) * height) + 10),
    )


def _measure_face(landmarks, width: int, height: int) -> GazeMeasurement | None:
    pose = _estimate_head_pose(landmarks, width, height)
    if pose is None or len(landmarks) <= 473:
        return None
    x1, y1, x2, y2 = _bbox(landmarks, width, height)
    area = ((x2 - x1) / width) * ((y2 - y1) / height)
    quality = max(0.0, min(1.0, area / 0.12))
    right = _iris_ratio(landmarks, 468, 33, 133, 159, 145)
    left = _iris_ratio(landmarks, 473, 362, 263, 386, 374)
    right_ear = _eye_aspect_ratio(landmarks, (33, 160, 158, 133, 153, 144))
    left_ear = _eye_aspect_ratio(landmarks, (362, 385, 387, 263, 373, 380))
    irises = (
        (int(landmarks[468].x * width), int(landmarks[468].y * height)),
        (int(landmarks[473].x * width), int(landmarks[473].y * height)),
    )
    return GazeMeasurement(
        pose[0], pose[1], (right[0] + left[0]) / 2.0,
        (right[1] + left[1]) / 2.0, (right_ear + left_ear) / 2.0,
        quality, (x1, y1, x2, y2), irises,
    )


class VisionMonitor:
    def __init__(self, config: Config):
        ensure_model(config)
        self.config = config
        options = mp.tasks.vision.FaceLandmarkerOptions(
            # Read via Python to support non-ASCII Windows user/folder names.
            base_options=mp.tasks.BaseOptions(model_asset_buffer=config.model_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.smoother = MeasurementSmoother(config.smoothing_alpha)
        self.calibrator = Calibrator(config)
        self.tracker = SustainedEventTracker()
        self.previous_center: tuple[float, float] | None = None
        self.missing_frames = 0

    def _select_primary(self, faces, width: int, height: int):
        if not faces:
            return None
        candidates = []
        for face in faces:
            box = _bbox(face, width, height)
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
            if self.previous_center is None:
                score = -area
            else:
                score = math.dist(center, self.previous_center) - math.sqrt(area) * 0.15
            candidates.append((score, face, center))
        _, selected, center = min(candidates, key=lambda item: item[0])
        self.previous_center = center
        return selected

    def process(self, rgb_frame: np.ndarray, width: int, height: int,
                timestamp_ms: int, now: float) -> VisionFrameResult:
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame), timestamp_ms
        )
        faces = result.face_landmarks or []
        selected = self._select_primary(faces, width, height)
        measurement = _measure_face(selected, width, height) if selected else None
        just_completed = False
        calibrated = None
        events: list[RiskEvent] = []

        if measurement is not None:
            self.missing_frames = 0
            measurement = self.smoother.update(measurement)
            was_ready = self.calibrator.ready
            calibration_result = self.calibrator.add(measurement, now)
            just_completed = not was_ready and self.calibrator.ready
            if calibration_result == "VERIFYING":
                events.append(RiskEvent(
                    now, "system", "CALIBRATION_VERIFYING", 0.0,
                    value=str(self.calibrator.attempt),
                    context="1차 기준 수집 완료; 정면 안정성 재검증",
                ))
            elif calibration_result not in {None, "READY"}:
                events.append(RiskEvent(
                    now, "system", "CALIBRATION_RETRY", 0.0,
                    value=str(self.calibrator.attempt), context=calibration_result,
                ))
            if self.calibrator.ready:
                calibrated = self.calibrator.apply(measurement)
        else:
            self.missing_frames += 1
            if self.missing_frames >= 3:
                self.smoother.reset()
                if not self.calibrator.ready and (
                    self.calibrator.started_at is not None
                    or self.calibrator.phase == "VERIFYING"
                ):
                    self.calibrator.reset("FACE LOST - HOLD STILL")
                    events.append(RiskEvent(
                        now, "system", "CALIBRATION_RETRY", 0.0,
                        value=str(self.calibrator.attempt),
                        context="얼굴 추적 중단으로 정면 기준 수집 재시작",
                    ))

        active: set[str] = set()
        if self.calibrator.ready:
            # A rejected pose does not mean that the detector lost the face.
            face_missing = not faces
            multiple = len(faces) > 1
            reliable = (
                calibrated is not None
                and calibrated.face_quality >= self.config.min_face_quality
            )
            head_away = reliable and calibrated.head_direction != "CENTER"
            gaze_away = (
                reliable and not calibrated.eyes_closed
                and calibrated.gaze_direction != "CENTER"
            )
            if head_away: active.add("HEAD_AWAY")
            if gaze_away: active.add("GAZE_AWAY")
            if face_missing: active.add("FACE_MISSING")
            if multiple: active.add("MULTIPLE_FACES")

            specs = [
                ("HEAD_AWAY", head_away, self.config.head_min_duration,
                 self.config.head_repeat_sec,
                 calibrated.head_direction if calibrated else "UNKNOWN",
                 calibrated.face_quality if calibrated else 0.0,
                 f"yaw={calibrated.yaw:.1f},pitch={calibrated.pitch:.1f}" if calibrated else ""),
                ("GAZE_AWAY", gaze_away, self.config.gaze_min_duration,
                 self.config.gaze_repeat_sec,
                 calibrated.gaze_direction if calibrated else "UNKNOWN",
                 calibrated.face_quality if calibrated else 0.0,
                 (f"gaze_dx={calibrated.gaze_dx:.3f},gaze_dy={calibrated.gaze_dy:.3f},"
                  f"ear={calibrated.ear:.3f}") if calibrated else ""),
                ("FACE_MISSING", face_missing, self.config.face_missing_min_duration,
                 self.config.face_missing_repeat_sec, "NOT_DETECTED", 1.0,
                 "캘리브레이션 완료 후 얼굴 미검출"),
                ("MULTIPLE_FACES", multiple, self.config.multiple_faces_min_duration,
                 self.config.multiple_faces_repeat_sec, str(len(faces)), 1.0,
                 "화면에서 복수 얼굴 검출"),
                ("FACE_TOO_SMALL", calibrated is not None and not reliable,
                 2.0, 15.0, f"{calibrated.face_quality:.3f}" if calibrated else "", 1.0,
                 "얼굴 영역이 작아 머리/시선 판정 제한; 실제 거리 측정 아님"),
                ("POSE_UNAVAILABLE", bool(faces) and measurement is None,
                 2.0, 15.0, "FACE_DETECTED", 1.0,
                 "얼굴은 검출됐으나 머리 자세 측정 불가; 부정행위 점수 없음"),
            ]
            for kind, enabled, minimum, repeat, value, confidence, context in specs:
                events.extend(self.tracker.update(
                    kind=kind, active=enabled, now=now, min_duration=minimum,
                    repeat_sec=repeat, weight=self.config.event_weights.get(kind, 0.0),
                    value=value, confidence=confidence, context=context,
                    release_grace_sec=(
                        self.config.head_release_grace_sec if kind == "HEAD_AWAY"
                        else self.config.gaze_release_grace_sec if kind == "GAZE_AWAY" else 0.0
                    ),
                ))

            for kind in ("HEAD_AWAY", "GAZE_AWAY"):
                state = self.tracker.states.get(kind, {})
                if state.get("emitted") and state.get("started") is not None:
                    active.add(kind)

        confirmed = {
            kind for kind in active
            if self.tracker.states.get(kind, {}).get("emitted", False)
        }
        durations = {
            kind: max(0.0, now - float(self.tracker.states[kind]["started"]))
            for kind in confirmed
        }
        head_state = self.tracker.states.get("HEAD_AWAY", {}) if self.calibrator.ready else {}
        head_started = head_state.get("started")
        head_duration = max(0.0, now - float(head_started)) if head_started is not None else 0.0
        if not self.calibrator.ready:
            head_status = "CALIBRATING"
        elif not faces:
            head_status = "NO_FACE"
        elif calibrated is None:
            head_status = "POSE_UNAVAILABLE"
        elif calibrated.face_quality < self.config.min_face_quality:
            head_status = "FACE_TOO_SMALL"
        elif calibrated.head_direction == "CENTER":
            head_status = "RETURN_GRACE" if "HEAD_AWAY" in confirmed else "CENTER"
        else:
            head_status = "CONFIRMED" if "HEAD_AWAY" in confirmed else "HOLD"

        return VisionFrameResult(
            calibrated=calibrated, events=events, active=active,
            confirmed_active=confirmed, durations=durations,
            face_count=len(faces),
            bbox=measurement.bbox if measurement else None,
            iris_points=measurement.iris_points if measurement else ((0, 0), (0, 0)),
            calibration_ready=self.calibrator.ready,
            calibration_progress=self.calibrator.progress(now),
            calibration_samples=len(self.calibrator.samples),
            calibration_baseline=dict(self.calibrator.baseline) if self.calibrator.ready else None,
            calibration_just_completed=just_completed,
            calibration_status=self.calibrator.status,
            head_status=head_status,
            head_duration_sec=head_duration,
        )

    def reset_calibration(self, reason: str = "MANUAL RECALIBRATION") -> None:
        self.calibrator.reset(reason)
        self.smoother.reset()
        self.tracker = SustainedEventTracker()
        self.previous_center = None
        self.missing_frames = 0

    def close(self) -> None:
        self.landmarker.close()
