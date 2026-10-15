from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _test_weights() -> dict[str, float]:
    return {
        "CTRL+C": 0.0,
        "CTRL+V": 0.0,
        "CTRL+X": 0.0,
        "ALT+TAB": 0.0,
        "WIN+TAB": 0.0,
        "SHIFT+INSERT": 0.0,
        "PRINT_SCREEN": 0.0,
        "WIN+SHIFT+S": 0.0,
        "LARGE_PASTE": 15.0,
        "TYPING_SPEED_SPIKE": 8.0,
        "TYPING_RHYTHM_ANOMALY": 8.0,
        "LONG_IDLE_THEN_BURST": 10.0,
        "WINDOW_FOCUS_LOST": 14.0,
        "HEAD_AWAY": 8.0,
        "GAZE_AWAY": 10.0,
        "FACE_MISSING": 12.0,
        "MULTIPLE_FACES": 20.0,
        "COMBINED_BEHAVIOR": 12.0,
    }


def _production_weights() -> dict[str, float]:
    return {
        "CTRL+C": 0.0,
        "CTRL+V": 0.0,
        "CTRL+X": 0.0,
        "ALT+TAB": 0.0,
        "WIN+TAB": 0.0,
        "SHIFT+INSERT": 0.0,
        "PRINT_SCREEN": 0.0,
        "WIN+SHIFT+S": 0.0,
        "LARGE_PASTE": 8.0,
        "TYPING_SPEED_SPIKE": 7.0,
        "TYPING_RHYTHM_ANOMALY": 6.0,
        "LONG_IDLE_THEN_BURST": 5.0,
        "WINDOW_FOCUS_LOST": 8.0,
        "HEAD_AWAY": 7.0,
        "GAZE_AWAY": 6.0,
        "FACE_MISSING": 10.0,
        "MULTIPLE_FACES": 20.0,
        "COMBINED_BEHAVIOR": 10.0,
    }


def _window_keywords() -> tuple[str, ...]:
    raw = os.environ.get(
        "EXAM_WINDOW_KEYWORDS", "Multimodal Anti-Cheating Monitor"
    )
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Config:
    test_mode: bool = field(default_factory=lambda: _env_flag(
        "ANTI_CHEATING_TEST_MODE", False
    ))
    use_webcam: bool = True
    camera_index: int = 0
    video_path: str = "test.mp4"
    model_path: Path = Path(os.environ.get(
        "FACE_LANDMARKER_MODEL", str(BASE_DIR / "face_landmarker.task")
    ))
    log_path: Path = Path(os.environ.get(
        "ANTI_CHEATING_LOG", str(BASE_DIR / "anti_cheating_events.csv")
    ))

    calibration_sec: float = 4.0
    calibration_min_samples: int = 45
    calibration_verify_sec: float = 1.5
    calibration_verify_min_samples: int = 20
    calibration_min_pass_ratio: float = 0.80
    calibration_max_abs_yaw: float = 30.0
    calibration_max_abs_pitch: float = 18.0
    calibration_max_yaw_mad: float = 4.5
    calibration_max_pitch_mad: float = 3.5
    calibration_max_gaze_mad: float = 0.045
    calibration_verify_yaw: float = 12.0
    calibration_verify_pitch: float = 8.0
    calibration_verify_gaze: float = 0.10
    min_face_quality: float = 0.30
    smoothing_alpha: float = 0.20
    # Tolerate brief direction flicker, without changing initial event thresholds.
    head_release_grace_sec: float = 0.50
    gaze_release_grace_sec: float = 0.30

    # 화면 HEAD 방향 기준. 정면 보정 후 yaw: 양수=LEFT, 음수=RIGHT.
    # 두 값 모두 양수로 설정합니다. 작을수록 해당 방향을 민감하게 감지합니다.
    head_left_yaw_threshold: float = 30.0
    head_right_yaw_threshold: float = 85.0
    head_pitch_threshold: float = 25.0
    gaze_x_threshold: float = 0.16
    gaze_y_threshold: float = 0.14
    eye_closed_ratio: float = 0.62

    head_min_duration: float = 1.5
    head_repeat_sec: float = 4.0
    gaze_min_duration: float = 1.0
    gaze_repeat_sec: float = 3.0
    face_missing_min_duration: float = 2.0
    face_missing_repeat_sec: float = 5.0
    multiple_faces_min_duration: float = 1.0
    multiple_faces_repeat_sec: float = 5.0

    # Development thresholds; not calibrated accuracy/probability estimates.
    score_window_sec: float = 20.0
    score_half_life_sec: float = 9.0
    review_threshold: float = 100.0
    combo_window_sec: float = 5.0
    combo_cooldown_sec: float = 4.0
    watch_enter: float = 25.0
    watch_exit: float = 14.0
    suspicious_enter: float = 50.0
    suspicious_exit: float = 34.0

    shortcut_cooldown_sec: float = 0.75
    # Provisional engineering reference, NOT a measured population average.
    typing_reference_kpm: float = 300.0
    typing_personal_max_weight: float = 0.35
    typing_speed_ratio: float = 2.2
    typing_speed_floor_kpm: float = 450.0
    typing_window_sec: float = 20.0
    typing_analysis_interval_sec: float = 5.0
    typing_event_cooldown_sec: float = 15.0
    large_paste_chars: int = 80
    long_idle_sec: float = 30.0
    burst_window_sec: float = 3.0
    burst_min_keys: int = 15

    exam_window_keywords: tuple[str, ...] = field(default_factory=_window_keywords)
    focus_lost_min_duration: float = 2.0
    focus_lost_repeat_sec: float = 10.0
    window_poll_sec: float = 0.25
    clipboard_poll_sec: float = 0.25
    event_display_sec: float = 6.0

    event_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_weights:
            profile = _test_weights() if self.test_mode else _production_weights()
            object.__setattr__(self, "event_weights", profile)
        required = {'LARGE_PASTE', 'WINDOW_FOCUS_LOST', 'COMBINED_BEHAVIOR',
                    'HEAD_AWAY', 'GAZE_AWAY', 'FACE_MISSING', 'MULTIPLE_FACES',
                    'TYPING_SPEED_SPIKE', 'TYPING_RHYTHM_ANOMALY', 'LONG_IDLE_THEN_BURST'}
        missing = required - self.event_weights.keys()
        if missing:
            raise ValueError(f'필수 이벤트 가중치 누락: {sorted(missing)}')
        import math
        if any(not math.isfinite(v) or v < 0 for v in self.event_weights.values()):
            raise ValueError('가중치는 유한한 0 이상의 수여야 합니다.')
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.review_threshold, self.score_window_sec, self.score_half_life_sec)):
            raise ValueError('점수 시간 및 검토 임계값은 유한한 양수여야 합니다.')
        if not self.test_mode:
            object.__setattr__(self, "head_min_duration", 2.0)
            object.__setattr__(self, "head_repeat_sec", 8.0)
            object.__setattr__(self, "gaze_x_threshold", 0.10)
            object.__setattr__(self, "gaze_y_threshold", 0.18)
            object.__setattr__(self, "gaze_min_duration", 1.5)
            object.__setattr__(self, "gaze_repeat_sec", 6.0)
            object.__setattr__(self, "face_missing_min_duration", 2.5)
            object.__setattr__(self, "face_missing_repeat_sec", 12.0)
            object.__setattr__(self, "multiple_faces_min_duration", 1.5)
            object.__setattr__(self, "multiple_faces_repeat_sec", 15.0)

    @property
    def effective_long_idle_sec(self) -> float:
        return 5.0 if self.test_mode else self.long_idle_sec

    @property
    def effective_burst_min_keys(self) -> int:
        return 8 if self.test_mode else self.burst_min_keys

    def large_paste_weight(self, chars: int) -> float:
        if chars < self.large_paste_chars:
            return 0.0
        base = self.event_weights['LARGE_PASTE']
        if self.test_mode:
            return base
        return base * (2.25 if chars >= 500 else 1.5 if chars >= 200 else 1.0)


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
