from __future__ import annotations

import queue
import sys
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activity_monitor import ClipboardChangeMonitor, WindowActivityMonitor, clipboard_metadata
from config import Config
from keyboard_monitor import KeyboardMonitor, keyboard
from models import GazeMeasurement, RiskEvent
from main import _ignore_console_ctrl_c
from risk_engine import RiskEngine
from vision_monitor import Calibrator, MeasurementSmoother, VisionMonitor
from types import SimpleNamespace
import numpy as np


class FakeKey:
    def __init__(self, char: str):
        self.char = char


class KeyboardTests(unittest.TestCase):
    def make_monitor(self):
        event_queue = queue.Queue()
        stop = threading.Event()
        return KeyboardMonitor(event_queue, stop, Config()), event_queue, stop

    def test_q_does_not_exit(self):
        monitor, _, stop = self.make_monitor()
        key = FakeKey("q")
        monitor._on_press(key)
        monitor._on_release(key)
        self.assertFalse(stop.is_set())

    def test_console_ctrl_c_handler_does_not_raise(self):
        _ignore_console_ctrl_c(None, None)

    def test_plain_escape_exits(self):
        monitor, _, stop = self.make_monitor()
        monitor._on_press(keyboard.Key.esc)
        monitor._on_release(keyboard.Key.esc)
        self.assertTrue(stop.is_set())

    def test_control_character_shortcut(self):
        monitor, event_queue, _ = self.make_monitor()
        monitor._on_press(keyboard.Key.ctrl_l)
        key = FakeKey("\x03")
        monitor._on_press(key)
        self.assertEqual(event_queue.get_nowait().kind, "CTRL+C")

    def test_task_manager_shortcut_does_not_trigger_plain_escape_exit(self):
        monitor, event_queue, stop = self.make_monitor()
        monitor._on_press(keyboard.Key.ctrl_l)
        monitor._on_press(keyboard.Key.shift_l)
        monitor._on_press(keyboard.Key.esc)
        self.assertTrue(event_queue.empty())
        monitor._on_release(keyboard.Key.esc)
        self.assertFalse(stop.is_set())

    def test_clipboard_metadata_never_returns_text(self):
        metadata = clipboard_metadata()
        self.assertEqual(set(metadata), {"type", "chars", "lines"})

    def test_large_paste_is_classified_without_storing_content(self):
        monitor, event_queue, _ = self.make_monitor()
        monitor._on_press(keyboard.Key.ctrl_l)
        with patch(
            "keyboard_monitor.clipboard_metadata",
            return_value={"type": "TEXT", "chars": 120, "lines": 4},
        ):
            monitor._on_press(FakeKey("\x16"))
        events = [event_queue.get_nowait() for _ in range(event_queue.qsize())]
        kinds = [event.kind for event in events]
        self.assertEqual(kinds, ["CTRL+V", "LARGE_PASTE"])
        self.assertEqual(events[0].weight, 0.0)
        self.assertEqual(events[1].weight, 8.0)

    def test_shift_insert_is_treated_as_paste(self):
        monitor, event_queue, _ = self.make_monitor()
        monitor._on_press(keyboard.Key.shift_l)
        with patch(
            "keyboard_monitor.clipboard_metadata",
            return_value={"type": "TEXT", "chars": 120, "lines": 2},
        ):
            monitor._on_press(keyboard.Key.insert)
        kinds = [event_queue.get_nowait().kind for _ in range(event_queue.qsize())]
        self.assertEqual(kinds, ["SHIFT+INSERT", "LARGE_PASTE"])

    def test_typing_speed_anomaly_uses_aggregate_timing_only(self):
        monitor, event_queue, _ = self.make_monitor()
        now = time.monotonic()
        monitor.started_at = now - 60.0
        monitor.baseline_ready = True
        monitor.baseline = {
            "kpm": 10.0, "interval_cv": 0.5,
            "mean_interval": 0.4, "mean_dwell": 0.1,
        }
        monitor.key_times = deque(now - 10.0 + index * 0.04 for index in range(200))
        monitor.intervals = deque(
            (now - 10.0 + index * 0.04, 0.04 + (index % 3) * 0.001)
            for index in range(199)
        )
        monitor._analyze(now)
        kinds = [event_queue.get_nowait().kind for _ in range(event_queue.qsize())]
        self.assertIn("TYPING_SPEED_SPIKE", kinds)


class RiskEngineTests(unittest.TestCase):
    def test_visual_and_input_create_combo(self):
        config = Config()
        engine = RiskEngine(config)
        engine.add(RiskEvent(100.0, "vision", "GAZE_AWAY", 10.0))
        accepted, _ = engine.add(RiskEvent(101.0, "window", "WINDOW_FOCUS_LOST", 8.0))
        self.assertIn("COMBINED_BEHAVIOR", [event.kind for event in accepted])

    def test_large_score_drop_can_return_to_normal(self):
        engine = RiskEngine(Config())
        engine.state = "SUSPICIOUS"
        engine.score = 0.0
        self.assertEqual(engine._transition(), ("SUSPICIOUS", "NORMAL"))

    def test_production_mode_deduplicates_window_shortcut(self):
        config = Config(test_mode=False)
        engine = RiskEngine(config)
        engine.add(RiskEvent(100.0, "keyboard", "ALT+TAB", 4.0))
        accepted, _ = engine.add(RiskEvent(
            102.0, "window", "WINDOW_FOCUS_LOST", 8.0
        ))
        self.assertEqual(accepted[0].weight, 8.0)
        self.assertAlmostEqual(engine.score, 8.0, delta=0.7)

    def test_test_mode_keeps_sensitive_window_scores(self):
        config = Config(test_mode=True)
        engine = RiskEngine(config)
        engine.add(RiskEvent(100.0, "keyboard", "ALT+TAB", 12.0))
        accepted, _ = engine.add(RiskEvent(
            102.0, "window", "WINDOW_FOCUS_LOST", 14.0
        ))
        self.assertEqual(accepted[0].weight, 14.0)


class VisionUtilityTests(unittest.TestCase):
    @staticmethod
    def measurement(yaw=0.0, gaze=0.0):
        return GazeMeasurement(
            yaw, 0.0, gaze, gaze, 0.3, 1.0,
            (0, 0, 10, 10), ((2, 2), (8, 2)),
        )

    def test_smoothing(self):
        smoother = MeasurementSmoother(0.2)
        smoother.update(self.measurement())
        value = smoother.update(self.measurement(yaw=10.0, gaze=1.0))
        self.assertAlmostEqual(value.raw_yaw, 2.0)
        self.assertAlmostEqual(value.gaze_x, 0.2)

    def test_calibration(self):
        config = Config(
            calibration_sec=0.0, calibration_min_samples=1,
            calibration_verify_sec=0.0, calibration_verify_min_samples=1,
        )
        calibrator = Calibrator(config)
        now = time.monotonic()
        self.assertEqual(calibrator.add(self.measurement(), now), "VERIFYING")
        self.assertEqual(calibrator.add(self.measurement(), now + 0.01), "READY")
        self.assertTrue(calibrator.ready)
        self.assertEqual(calibrator.apply(self.measurement()).head_direction, "CENTER")

    def test_off_center_calibration_is_rejected(self):
        config = Config(calibration_sec=0.0, calibration_min_samples=1)
        calibrator = Calibrator(config)
        result = calibrator.add(self.measurement(yaw=-32.0), time.monotonic())
        self.assertEqual(result, "FACE NOT CENTERED HORIZONTALLY")
        self.assertFalse(calibrator.ready)

    def test_unstable_verification_restarts_calibration(self):
        config = Config(
            calibration_sec=0.0, calibration_min_samples=1,
            calibration_verify_sec=0.0, calibration_verify_min_samples=1,
        )
        calibrator = Calibrator(config)
        now = time.monotonic()
        calibrator.add(self.measurement(), now)
        result = calibrator.add(self.measurement(yaw=20.0), now + 0.01)
        self.assertTrue(str(result).startswith("VERIFY FAILED"))
        self.assertEqual(calibrator.phase, "COLLECTING")

    def test_calibration_subtracts_personal_baseline(self):
        config = Config(
            calibration_sec=0.0, calibration_min_samples=1,
            calibration_verify_sec=0.0, calibration_verify_min_samples=1,
        )
        calibrator = Calibrator(config)
        calibrator.add(self.measurement(yaw=10.0, gaze=0.4), 100.0)
        calibrator.add(self.measurement(yaw=10.0, gaze=0.4), 100.1)
        value = calibrator.apply(self.measurement(yaw=12.0, gaze=0.43))
        self.assertAlmostEqual(value.yaw, 2.0)
        self.assertAlmostEqual(value.gaze_dx, 0.03)
        self.assertEqual(value.head_direction, "CENTER")
        self.assertEqual(value.gaze_direction, "CENTER")

    def test_missing_face_restarts_unfinished_calibration(self):
        monitor = VisionMonitor.__new__(VisionMonitor)
        monitor.config = Config()
        monitor.landmarker = SimpleNamespace(
            detect_for_video=lambda *_: SimpleNamespace(face_landmarks=[])
        )
        monitor.smoother = MeasurementSmoother(0.2)
        monitor.calibrator = Calibrator(monitor.config)
        monitor.calibrator.add(self.measurement(), 100.0)
        monitor.previous_center = None
        monitor.missing_frames = 0
        for i in range(3):
            result = monitor.process(np.zeros((10, 10, 3), dtype=np.uint8),
                                     10, 10, i, 101.0 + i)
        self.assertFalse(monitor.calibrator.ready)
        self.assertIsNone(monitor.calibrator.started_at)
        self.assertEqual(monitor.calibrator.samples, [])
        self.assertEqual([e.kind for e in result.events], ["CALIBRATION_RETRY"])
        # Further absent frames must not emit repeated calibration resets.
        result = monitor.process(np.zeros((10, 10, 3), dtype=np.uint8),
                                 10, 10, 4, 105.0)
        self.assertEqual(result.events, [])


class WindowTests(unittest.TestCase):
    def test_exam_window_matching(self):
        config = Config(exam_window_keywords=("exam room",))
        monitor = WindowActivityMonitor(queue.Queue(), threading.Event(), config)
        self.assertTrue(monitor._is_exam_window("Student - Exam Room"))
        self.assertFalse(monitor._is_exam_window("Search results"))

    def test_clipboard_sequence_number_is_numeric(self):
        monitor = ClipboardChangeMonitor(queue.Queue(), threading.Event(), Config())
        self.assertIsInstance(monitor._sequence_number(), int)


class TestModeTests(unittest.TestCase):
    def test_test_mode_relaxes_idle_and_burst_conditions(self):
        config = Config(test_mode=True)
        self.assertEqual(config.effective_long_idle_sec, 5.0)
        self.assertEqual(config.effective_burst_min_keys, 8)

    def test_normal_mode_keeps_production_conditions(self):
        config = Config(test_mode=False, head_pitch_threshold=25.0)
        self.assertEqual(config.effective_long_idle_sec, 30.0)
        self.assertEqual(config.effective_burst_min_keys, 15)
        self.assertEqual(config.event_weights["CTRL+C"], 0.0)
        self.assertEqual(config.event_weights["GAZE_AWAY"], 6.0)
        self.assertEqual(config.head_pitch_threshold, 25.0)
        self.assertEqual(config.gaze_min_duration, 1.5)

    def test_test_mode_keeps_sensitive_profile(self):
        config = Config(test_mode=True, head_pitch_threshold=10.0)
        self.assertEqual(config.event_weights["CTRL+C"], 0.0)
        self.assertEqual(config.event_weights["GAZE_AWAY"], 10.0)
        self.assertEqual(config.head_pitch_threshold, 10.0)
        self.assertEqual(config.gaze_min_duration, 1.0)


if __name__ == "__main__":
    unittest.main()
