from __future__ import annotations

import queue
import sys
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from models import RiskEvent, GazeMeasurement
from risk_engine import RiskEngine
from keyboard_monitor import KeyboardMonitor
from typing_setup import make_reference, TypingReference, run_typing_setup
from vision_monitor import VisionMonitor, MeasurementSmoother, Calibrator, SustainedEventTracker
from activity_monitor import WindowActivityMonitor


class ReferenceTests(unittest.TestCase):
    def test_extreme_samples_have_bounded_influence(self):
        cfg = Config(test_mode=False)
        fast = make_reference(cfg, [i * 0.025 for i in range(300)], 12)
        slow = make_reference(cfg, [i * 0.9 for i in range(100)], 90)
        self.assertTrue(fast.valid)
        self.assertTrue(slow.valid)
        self.assertLessEqual(fast.reference_kpm, 326.25)
        self.assertGreaterEqual(slow.reference_kpm, 273.75)
        self.assertLessEqual(fast.personal_weight, 0.35)

    def test_paste_or_short_sample_uses_default(self):
        cfg = Config()
        for times, elapsed, paste in [([i * .1 for i in range(90)], 9, True),
                                      ([1, 2], 8, False)]:
            ref = make_reference(cfg, times, elapsed, paste)
            self.assertFalse(ref.valid)
            self.assertEqual(ref.reference_kpm, cfg.typing_reference_kpm)

    def test_reading_pause_is_excluded(self):
        cfg = Config()
        times = [i * .15 for i in range(60)] + [20 + i * .15 for i in range(60)]
        ref = make_reference(cfg, times, 30)
        self.assertAlmostEqual(ref.personal_kpm, 400)

    def test_exam_typing_cannot_relearn_reference(self):
        monitor = KeyboardMonitor(queue.Queue(), threading.Event(), Config())
        ref = TypingReference(300, 400, .2, .5, 100, 20, True)
        monitor.set_typing_reference(ref)
        monitor.started_at = 1
        monitor.key_times = deque(100 + i * .04 for i in range(200))
        monitor._analyze(120)
        self.assertEqual(monitor.baseline['kpm'], 300)


class RiskTests(unittest.TestCase):
    def test_typing_alone_never_scores_or_creates_combo(self):
        engine = RiskEngine(Config())
        for t in [100, 116, 132]:
            accepted, _ = engine.add(RiskEvent(t, 'typing', 'TYPING_SPEED_SPIKE', 7))
            self.assertEqual(accepted[0].weight, 0)
            self.assertEqual(engine.score, 0)
            self.assertEqual(len(accepted), 1)

    def test_supported_typing_is_only_two_points(self):
        engine = RiskEngine(Config())
        engine.set_active_visual({'HEAD_AWAY'})
        accepted, _ = engine.add(RiskEvent(100, 'typing', 'TYPING_SPEED_SPIKE', 7))
        self.assertEqual(accepted[0].weight, 2)
        self.assertEqual(len(accepted), 1)

    def test_copy_stays_zero_even_with_confirmed_visual(self):
        engine = RiskEngine(Config())
        engine.set_active_visual(set())
        accepted, _ = engine.add(RiskEvent(100, 'keyboard', 'CTRL+C', 2))
        self.assertEqual(len(accepted), 1)
        engine.set_active_visual({'HEAD_AWAY'})
        accepted, _ = engine.add(RiskEvent(105, 'keyboard', 'CTRL+C', 2))
        self.assertNotIn('COMBINED_BEHAVIOR', [e.kind for e in accepted])
        self.assertEqual(engine.total_points, 0)

    def test_sustained_duration_alone_does_not_add_unlogged_points(self):
        engine = RiskEngine(Config())
        engine.set_sustained({'FACE_MISSING': 9.9})
        engine.refresh(100)
        self.assertEqual(engine.score, 0)
        engine.set_sustained({'FACE_MISSING': 10})
        engine.refresh(110)
        self.assertEqual(engine.state, 'NORMAL')
        self.assertEqual(engine.score, 0)
        engine.set_sustained({'FACE_MISSING': 30})
        engine.refresh(130)
        self.assertEqual(engine.state, 'NORMAL')
        self.assertEqual(engine.score, 0)
        engine.refresh(300)
        self.assertEqual(engine.score, 0)
        engine.set_sustained({})
        engine.refresh(301)
        self.assertEqual(engine.state, 'NORMAL')

    def test_sustained_durations_do_not_double_count_events(self):
        engine = RiskEngine(Config())
        engine.set_sustained({'HEAD_AWAY': 40, 'GAZE_AWAY': 40})
        engine.refresh(100)
        self.assertEqual(engine.score, 0)

    def test_repeated_window_and_multiple_face_events_accumulate(self):
        for kind, first, second in [('WINDOW_FOCUS_LOST', 10, 30),
                                    ('MULTIPLE_FACES', 5, 15)]:
            engine = RiskEngine(Config())
            engine.set_sustained({kind: first})
            engine.add(RiskEvent(100, 'vision', kind, 25))
            engine.refresh(100)
            self.assertEqual(engine.state, 'WATCH')
            engine.set_sustained({kind: second})
            engine.add(RiskEvent(200, 'vision', kind, 25))
            engine.refresh(200)
            self.assertEqual(engine.state, 'WATCH')
            self.assertEqual(engine.total_points, 50)

    def test_delayed_event_does_not_keep_expired_events(self):
        engine = RiskEngine(Config())
        engine.add(RiskEvent(110, 'vision', 'HEAD_AWAY', 2))
        engine.add(RiskEvent(100, 'vision', 'HEAD_AWAY', 2))
        engine.refresh(131)
        self.assertEqual(list(engine.events), [])
        self.assertEqual(engine.score, 0)
        self.assertEqual(engine.total_points, 4)


class IntegrationTests(unittest.TestCase):
    def test_main_connects_setup_and_cleanup_without_hidden_floor(self):
        import main as app
        from models import VisionFrameResult
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap = SimpleNamespace(isOpened=lambda: True, get=lambda _: 30,
                              read=iter([(True, frame), (False, None)]).__next__,
                              release=lambda: released.append('camera'))
        released, rows = [], []
        vision = SimpleNamespace(process=lambda *_: VisionFrameResult(
            calibration_ready=True, confirmed_active={'FACE_MISSING'},
            durations={'FACE_MISSING': 30}), close=lambda: released.append('vision'))
        logger = SimpleNamespace(write=lambda event, score, state: rows.append((event, score, state)))
        window = SimpleNamespace(start=lambda: None, join=lambda: released.append('window'),
                                 focus_lost=False, focus_lost_since=None)
        clipboard = SimpleNamespace(start=lambda: None, join=lambda: released.append('clipboard'))
        with patch.object(app, 'run_typing_setup', return_value=TypingReference(300)), \
             patch.object(app, 'EventLogger', return_value=logger), \
             patch.object(app, 'VisionMonitor', return_value=vision), \
             patch.object(app, 'WindowActivityMonitor', return_value=window), \
             patch.object(app, 'ClipboardChangeMonitor', return_value=clipboard), \
             patch.object(app.KeyboardMonitor, 'start'), \
             patch.object(app.KeyboardMonitor, 'join'), \
             patch.object(app.ExitMonitor, 'start'), \
             patch.object(app.ExitMonitor, 'join'), \
             patch.object(app.cv2, 'VideoCapture', return_value=cap), \
             patch.object(app.cv2, 'namedWindow'), patch.object(app.cv2, 'imshow'), \
             patch.object(app.cv2, 'destroyAllWindows'), \
             patch.object(app.cv2, 'waitKey', return_value=-1), \
             patch.object(app, 'render'), patch.object(app, 'print_summary'), \
             patch.object(app.signal, 'signal'):
            app.main()
        self.assertFalse(any(row[0].kind == 'SUSTAINED_LEVEL' for row in rows))
        self.assertEqual(rows[-1][1:], (0, 'NORMAL'))
        self.assertEqual(set(released), {'camera', 'vision', 'window', 'clipboard'})

    def test_three_second_burst_detected_between_analysis_ticks(self):
        q = queue.Queue()
        monitor = KeyboardMonitor(q, threading.Event(), Config())
        monitor.last_typing_at = 60
        for i in range(15):
            key = type('TestKey', (), {'char': 'a'})()
            monitor._record_typing_key(key, 100 + i * .1)
        events = [q.get_nowait() for _ in range(q.qsize())]
        self.assertEqual([e.kind for e in events], ['LONG_IDLE_THEN_BURST'])

    def test_vision_confirms_only_after_duration(self):
        cfg = Config(test_mode=False)
        monitor = VisionMonitor.__new__(VisionMonitor)
        monitor.config = cfg
        monitor.landmarker = SimpleNamespace(
            detect_for_video=lambda *_: SimpleNamespace(face_landmarks=[]))
        monitor.smoother = MeasurementSmoother(.2)
        monitor.calibrator = Calibrator(cfg)
        monitor.calibrator.ready = True
        monitor.tracker = SustainedEventTracker()
        monitor.previous_center = None
        monitor.missing_frames = 0
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        first = monitor.process(image, 10, 10, 0, 100)
        self.assertEqual(first.active, {'FACE_MISSING'})
        self.assertEqual(first.confirmed_active, set())
        second = monitor.process(image, 10, 10, 2500, 102.5)
        self.assertEqual(second.confirmed_active, {'FACE_MISSING'})
        self.assertEqual(second.durations['FACE_MISSING'], 2.5)

    def test_window_duration_clears_when_focus_returns(self):
        # Real monitor loop, mocked window sequence and clock; no desktop changes.
        stop = threading.Event()
        stop.wait = lambda _: next(waits)
        waits = iter([False, False, False, False, True])
        monitor = WindowActivityMonitor(queue.Queue(), stop,
                                       Config(exam_window_keywords=('exam',)))
        with patch('activity_monitor._foreground_window', side_effect=[
            ('exam', 'exam.exe'), ('other', 'other.exe'),
            ('other', 'other.exe'), ('exam', 'exam.exe')]), \
             patch('activity_monitor.time.monotonic', side_effect=[100, 101, 104, 105]):
            monitor._run()
        self.assertIsNone(monitor.focus_lost_since)
        self.assertFalse(monitor.focus_lost)
        events = [monitor.output_queue.get_nowait()
                  for _ in range(monitor.output_queue.qsize())]
        self.assertEqual([e.kind for e in events],
                         ['WINDOW_FOCUS_LOST', 'WINDOW_FOCUS_RESTORED'])


if __name__ == '__main__':
    unittest.main()
