from __future__ import annotations

import cv2

from config import Config
from models import VisionFrameResult


WINDOW_TITLE = "Multimodal Anti-Cheating Monitor"


def _text(frame, value: str, position, color, scale=0.65, thickness=2) -> None:
    cv2.putText(
        frame, value, position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness, cv2.LINE_AA,
    )


def _risk_color(state: str):
    return {
        "NORMAL": (0, 210, 0),
        "WATCH": (0, 190, 255),
        "SUSPICIOUS": (0, 0, 255),
        "REVIEW_REQUIRED": (0, 0, 255),
    }[state]


def _event_overlay(frame, recent_events, now: float, config: Config,
                   active_warnings: list[str]) -> None:
    if not config.test_mode:
        return
    height, width = frame.shape[:2]
    panel_width = min(430, max(280, width // 2))
    x1, x2 = width - panel_width - 10, width - 10
    visible = [
        event for event in recent_events
        if now - event.timestamp <= config.event_display_sec
    ][-5:]
    rows = 2 + len(active_warnings) + len(visible)
    panel_height = min(height - 20, 34 + rows * 25)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, 10), (x2, 10 + panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    _text(frame, "TEST MODE - LIVE EVENTS", (x1 + 12, 36), (0, 220, 255), 0.58)
    y = 62
    for warning in active_warnings:
        _text(frame, f"ACTIVE: {warning}", (x1 + 12, y), (0, 0, 255), 0.52)
        y += 25
    for event in reversed(visible):
        age = now - event.timestamp
        label = f"{event.kind}  +{event.weight:.0f}  ({age:.1f}s)"
        color = (0, 80, 255) if event.weight > 0 else (180, 180, 180)
        _text(frame, label, (x1 + 12, y), color, 0.48, 1)
        y += 25


def _head_status_text(vision: VisionFrameResult, config: Config) -> str:
    messages = {
        "CALIBRATING": "WAIT - finish calibration",
        "NO_FACE": "WAIT - face not detected",
        "POSE_UNAVAILABLE": "WAIT - head angle unavailable",
        "FACE_TOO_SMALL": "WAIT - move closer",
        "CENTER": "CENTER - below angle limits",
        "HOLD": f"HOLD {vision.head_duration_sec:.1f}/{config.head_min_duration:.1f}s",
        "CONFIRMED": f"ACTIVE {vision.head_duration_sec:.1f}s (repeat {config.head_repeat_sec:g}s)",
        "RETURN_GRACE": "RETURNING - clearing previous head event",
    }
    return "Head scoring: " + messages.get(vision.head_status, vision.head_status)


def _head_diagnostics(frame, vision: VisionFrameResult, config: Config) -> None:
    limits = (
        f"HEAD limits: LEFT >= {config.head_left_yaw_threshold:g} | "
        f"RIGHT <= -{config.head_right_yaw_threshold:g} | "
        f"Pitch +/-{config.head_pitch_threshold:g}"
    )
    for text, y in ((limits, 204), (_head_status_text(vision, config), 228)):
        scale = 0.48
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        scale *= min(1.0, max(1, frame.shape[1] - 40) / max(1, text_width))
        _text(frame, text, (20, y), (255, 220, 150), scale, 1)


def render(frame, vision: VisionFrameResult, engine, keyboard_monitor,
           window_monitor, recent_events, now: float, config: Config) -> None:
    height = frame.shape[0]
    if vision.bbox:
        x1, y1, x2, y2 = vision.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
        for iris in vision.iris_points:
            cv2.circle(frame, iris, 4, (255, 0, 255), -1)

    if not vision.calibration_ready:
        _text(
            frame,
            f"Calibration: {vision.calibration_progress * 100:.0f}% - Look forward",
            (20, 40), (0, 190, 255), 0.75,
        )
        _text(
            frame, f"Valid samples: {vision.calibration_samples}",
            (20, 75), (255, 255, 255), 0.6,
        )
        _text(
            frame, vision.calibration_status,
            (20, 108), (0, 190, 255), 0.52,
        )
    elif vision.calibrated:
        item = vision.calibrated
        _text(
            frame,
            f"Head: {item.head_direction} Yaw {item.yaw:.1f} Pitch {item.pitch:.1f}",
            (20, 40), (0, 0, 255) if "HEAD_AWAY" in vision.active else (0, 220, 0),
        )
        _text(
            frame,
            f"Gaze: {item.gaze_direction} dX {item.gaze_dx:.2f} dY {item.gaze_dy:.2f} EAR {item.ear:.2f}",
            (20, 75), (0, 0, 255) if "GAZE_AWAY" in vision.active else (0, 220, 0),
            0.58,
        )
        _text(
            frame, f"Face size score: {item.face_quality:.2f} Faces: {vision.face_count}",
            (20, 108), (255, 255, 0), 0.55,
        )
    else:
        label = "Face detected - pose unavailable" if vision.face_count else "Face Not Detected"
        _text(frame, label, (20, 40), (0, 190, 255) if vision.face_count else (0, 0, 255), 0.65)

    _text(
        frame, f"R: {engine.score:.1f} | C: {engine.total_points:.1f} | B: {engine.reviewed_points:.1f}",
        (20, 145), _risk_color(engine.state), 0.78,
    )
    baseline = "PERSONAL ASSIST" if keyboard_monitor.typing_reference.valid else "DEFAULT"
    if keyboard_monitor.hook_failed:
        _text(frame, "Keyboard monitoring stopped - restart application",
              (20, 252), (0, 190, 255), 0.48, 1)
    _text(
        frame, f"Typing baseline: {baseline} | Focus monitor: {'ARMED' if window_monitor.armed else 'WAIT'}",
        (20, 178), (255, 255, 255), 0.52, 1,
    )
    _head_diagnostics(frame, vision, config)
    _text(
        frame, f"Unreviewed: {engine.unreviewed_points:.1f}/{config.review_threshold:g} | {'REVIEW PENDING' if engine.review_pending else 'MONITORING'} | Exam continues",
        (20, height - 43), (190, 190, 190), 0.46, 1,
    )
    active_warnings = sorted(vision.active)
    if window_monitor.focus_lost:
        active_warnings.append("WINDOW_FOCUS_LOST")
    _event_overlay(frame, recent_events, now, config, active_warnings)
    _text(frame, "Recalibrate: R | Exit: ESC or close window", (20, height - 20),
          (210, 210, 210), 0.5, 1)
