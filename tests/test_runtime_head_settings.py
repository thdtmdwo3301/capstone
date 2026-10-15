from __future__ import annotations
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from main import _runtime_settings
from models import GazeMeasurement
from risk_engine import RiskEngine
from vision_monitor import Calibrator, MeasurementSmoother, SustainedEventTracker, VisionMonitor


class RuntimeHeadSettingsTests(unittest.TestCase):
    @staticmethod
    def measurement(yaw=0.0, pitch=0.0, quality=1.0):
        return GazeMeasurement(yaw, pitch, .5, .5, .4, quality,
                               (0, 0, 10, 10), ((2, 2), (8, 2)))

    @staticmethod
    def monitor(config, ready=True, has_face=True):
        monitor = VisionMonitor.__new__(VisionMonitor)
        monitor.config = config
        face = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
        faces = [face] if has_face else []
        monitor.landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(face_landmarks=faces))
        monitor.smoother = MeasurementSmoother(1.0)
        monitor.calibrator = Calibrator(config)
        monitor.calibrator.ready = ready
        monitor.calibrator.baseline = dict(yaw=0.0, pitch=0.0, gaze_x=.5, gaze_y=.5, ear=.4)
        monitor.tracker = SustainedEventTracker()
        monitor.previous_center = None
        monitor.missing_frames = 0
        return monitor

    def test_explicit_pitch_limit_is_used_in_both_modes(self):
        for mode in (False, True):
            with self.subTest(mode=mode):
                config = Config(test_mode=mode, head_pitch_threshold=25.0)
                monitor = self.monitor(config)
                self.assertEqual(config.head_pitch_threshold, 25.0)
                self.assertEqual(monitor.calibrator.apply(self.measurement(pitch=20)).head_direction, 'CENTER')
                self.assertEqual(monitor.calibrator.apply(self.measurement(pitch=26)).head_direction, 'UP')
                self.assertEqual(monitor.calibrator.apply(self.measurement(pitch=-26)).head_direction, 'DOWN')

    def test_current_config_reaches_score_engine(self):
        config = Config(test_mode=False)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        samples = ((config.head_left_yaw_threshold, 'LEFT'),
                   (-config.head_right_yaw_threshold, 'RIGHT'))
        for yaw, expected in samples:
            with self.subTest(direction=expected):
                monitor = self.monitor(config)
                with patch('vision_monitor._measure_face', return_value=self.measurement(yaw)):
                    first = monitor.process(frame, 10, 10, 0, 100)
                    confirmed = monitor.process(frame, 10, 10, 2100, 102.1)
                self.assertEqual(first.head_status, 'HOLD')
                self.assertEqual(confirmed.head_status, 'CONFIRMED')
                self.assertEqual(confirmed.calibrated.head_direction, expected)
                engine = RiskEngine(config)
                for event in confirmed.events:
                    engine.add(event)
                self.assertGreater(engine.score, 0)
                print(f'CURRENT CONFIG: {expected} at yaw={yaw:g} -> HEAD_AWAY, score={engine.score:g}')

    def test_diagnostics_identify_why_head_is_not_scoring(self):
        cfg = Config()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cases = ((False, True, self.measurement(), 'CALIBRATING'),
                 (True, False, None, 'NO_FACE'),
                 (True, True, None, 'POSE_UNAVAILABLE'),
                 (True, True, self.measurement(quality=.1), 'FACE_TOO_SMALL'),
                 (True, True, self.measurement(), 'CENTER'))
        for ready, has_face, measurement, expected in cases:
            with self.subTest(status=expected):
                monitor = self.monitor(cfg, ready, has_face)
                with patch('vision_monitor._measure_face', return_value=measurement):
                    result = monitor.process(frame, 10, 10, 0, 100)
                self.assertEqual(result.head_status, expected)
                self.assertFalse(result.confirmed_active)

    def test_startup_summary_uses_effective_settings(self):
        cfg = Config(test_mode=False, head_left_yaw_threshold=81,
                     head_right_yaw_threshold=24.5, head_pitch_threshold=27)
        summary = _runtime_settings(cfg)
        for value in ('mode=NORMAL', 'head_left=81', 'head_right=24.5', 'head_pitch=27', 'head_hold=2s'):
            self.assertIn(value, summary)


if __name__ == '__main__':
    unittest.main()