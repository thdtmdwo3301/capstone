from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class RiskEvent:
    timestamp: float
    source: str
    kind: str
    weight: float
    value: str = ""
    duration: float = 0.0
    confidence: float = 1.0
    context: str = ""
    event_id: str = field(default_factory=lambda: uuid4().hex)
    related_ids: tuple[str, ...] = ()


@dataclass
class GazeMeasurement:
    raw_yaw: float
    raw_pitch: float
    gaze_x: float
    gaze_y: float
    ear: float
    face_quality: float
    bbox: tuple[int, int, int, int]
    iris_points: tuple[tuple[int, int], tuple[int, int]]


@dataclass
class CalibratedMeasurement:
    yaw: float
    pitch: float
    gaze_dx: float
    gaze_dy: float
    ear: float
    eyes_closed: bool
    face_quality: float
    head_direction: str
    gaze_direction: str


@dataclass
class TypingSnapshot:
    timestamp: float
    key_count: int
    keys_per_minute: float
    mean_interval: float
    interval_cv: float
    mean_dwell: float
    backspace_ratio: float
    idle_sec: float


@dataclass
class VisionFrameResult:
    calibrated: CalibratedMeasurement | None = None
    events: list[RiskEvent] = field(default_factory=list)
    active: set[str] = field(default_factory=set)
    confirmed_active: set[str] = field(default_factory=set)
    durations: dict[str, float] = field(default_factory=dict)
    face_count: int = 0
    bbox: tuple[int, int, int, int] | None = None
    iris_points: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
    calibration_ready: bool = False
    calibration_progress: float = 0.0
    calibration_samples: int = 0
    calibration_baseline: dict[str, float] | None = None
    calibration_just_completed: bool = False
    calibration_status: str = "LOOK FORWARD"
    head_status: str = "CALIBRATING"
    head_duration_sec: float = 0.0
