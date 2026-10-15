from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from models import GazeMeasurement
from risk_engine import RiskEngine
from vision_monitor import Calibrator, MeasurementSmoother, SustainedEventTracker, VisionMonitor


class HeadDirectionThresholdTests(unittest.TestCase):
    @staticmethod
    def measurement(yaw, pitch=0.0):
        return GazeMeasurement(yaw, pitch, .5, .5, .4, 1.0,
                               (0, 0, 10, 10), ((2, 2), (8, 2)))

    @staticmethod
    def calibrator(config, baseline_yaw=0.0):
        calibrator = Calibrator(config)
        calibrator.ready = True
        calibrator.baseline = dict(yaw=baseline_yaw, pitch=0.0,
                                   gaze_x=.5, gaze_y=.5, ear=.4)
        return calibrator

    def test_asymmetric_boundaries_in_both_modes(self):
        for test_mode in (False, True):
            cfg = Config(test_mode=test_mode, head_left_yaw_threshold=65.0,
                         head_right_yaw_threshold=15.0)
            calibrator = self.calibrator(cfg)
            for yaw, expected in ((0, 'CENTER'), (15, 'CENTER'), (64.9, 'CENTER'),
                                  (65, 'LEFT'), (70, 'LEFT'), (-14.9, 'CENTER'),
                                  (-15, 'RIGHT'), (-65, 'RIGHT')):
                with self.subTest(test_mode=test_mode, yaw=yaw):
                    self.assertEqual(calibrator.apply(self.measurement(yaw)).head_direction, expected)

    def test_each_side_can_be_adjusted_independently(self):
        left_changed = self.calibrator(Config(head_left_yaw_threshold=40, head_right_yaw_threshold=15))
        right_changed = self.calibrator(Config(head_left_yaw_threshold=65, head_right_yaw_threshold=25))
        self.assertEqual(left_changed.apply(self.measurement(45)).head_direction, 'LEFT')
        self.assertEqual(left_changed.apply(self.measurement(-15)).head_direction, 'RIGHT')
        self.assertEqual(right_changed.apply(self.measurement(-20)).head_direction, 'CENTER')
        self.assertEqual(right_changed.apply(self.measurement(65)).head_direction, 'LEFT')

    def test_thresholds_use_calibrated_not_raw_yaw(self):
        calibrator = self.calibrator(Config(head_left_yaw_threshold=65, head_right_yaw_threshold=15), baseline_yaw=10)
        for raw_yaw, corrected, expected in ((65, 55, 'CENTER'), (75, 65, 'LEFT'),
                                            (-4, -14, 'CENTER'), (-5, -15, 'RIGHT')):
            with self.subTest(raw_yaw=raw_yaw):
                result = calibrator.apply(self.measurement(raw_yaw))
                self.assertEqual(result.yaw, corrected)
                self.assertEqual(result.head_direction, expected)

    def test_pitch_and_gaze_still_work_with_centered_yaw(self):
        calibrator = self.calibrator(Config(test_mode=False, head_pitch_threshold=15.0))
        self.assertEqual(calibrator.apply(self.measurement(0, 16)).head_direction, 'UP')
        self.assertEqual(calibrator.apply(self.measurement(0, -16)).head_direction, 'DOWN')
        sample = self.measurement(0)
        sample.gaze_x = .8
        result = calibrator.apply(sample)
        self.assertEqual(result.head_direction, 'CENTER')
        self.assertEqual(result.gaze_direction, 'RIGHT')

    def test_each_direction_scores_only_after_required_duration(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        for yaw, direction in ((65, 'LEFT'), (-15, 'RIGHT')):
            with self.subTest(direction=direction):
                cfg = Config(test_mode=False, head_left_yaw_threshold=65, head_right_yaw_threshold=15)
                monitor = VisionMonitor.__new__(VisionMonitor)
                monitor.config = cfg
                face = [SimpleNamespace(x=.5, y=.5) for _ in range(478)]
                monitor.landmarker = SimpleNamespace(detect_for_video=lambda *_: SimpleNamespace(face_landmarks=[face]))
                monitor.smoother = MeasurementSmoother(1.0)
                monitor.calibrator = self.calibrator(cfg)
                monitor.tracker = SustainedEventTracker()
                monitor.previous_center = None
                monitor.missing_frames = 0
                with patch('vision_monitor._measure_face', return_value=self.measurement(yaw)):
                    first = monitor.process(image, 10, 10, 0, 100.0)
                    early = monitor.process(image, 10, 10, 1999, 101.999)
                    confirmed = monitor.process(image, 10, 10, 2000, 102.0)
                self.assertIn('HEAD_AWAY', first.active)
                self.assertEqual(first.head_status, 'HOLD')
                self.assertEqual(early.head_status, 'HOLD')
                self.assertAlmostEqual(early.head_duration_sec, 1.999)
                self.assertEqual(confirmed.head_status, 'CONFIRMED')
                self.assertFalse(first.confirmed_active)
                self.assertFalse(early.confirmed_active)
                self.assertFalse(first.events)
                self.assertFalse(early.events)
                self.assertEqual(confirmed.confirmed_active, {'HEAD_AWAY'})
                self.assertEqual([(e.kind, e.value) for e in confirmed.events], [('HEAD_AWAY', direction)])
                engine = RiskEngine(cfg)
                for event in confirmed.events:
                    engine.add(event)
                self.assertEqual(engine.score, cfg.event_weights['HEAD_AWAY'])


if __name__ == '__main__':
    unittest.main()