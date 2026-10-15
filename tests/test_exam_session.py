from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from risk_engine import RiskEngine
from models import RiskEvent, VisionFrameResult
from event_logger import EventLogger
from reviewer_data import read_snapshot
from exam_profiles import ExamProfile, load_profiles, save_profiles
from exam_process import WindowsJob, ExamApplication
from session_lock import SessionLock


class CumulativeTests(unittest.TestCase):
    def test_no_decay_or_expiry_and_only_new_session_resets(self):
        e = RiskEngine(Config())
        e.add(RiskEvent(100, 'vision', 'HEAD_AWAY', 7, confidence=.5))
        e.refresh(10000)
        self.assertEqual(e.total_points, 3.5)
        self.assertEqual(e.score, 0)
        self.assertFalse(e.events)
        e.set_active_visual(set()); e.set_sustained({}); e.refresh(20000)
        self.assertEqual(e.total_points, 3.5)
        self.assertEqual(e.score, 0)
        self.assertEqual(RiskEngine(Config()).score, 0)

    def test_exact_threshold_and_latched_state(self):
        e = RiskEngine(Config())
        for i in range(3): e.add(RiskEvent(100 + 30*i, 'vision', 'MULTIPLE_FACES', 25))
        self.assertEqual(e.total_points, 75)
        e.add(RiskEvent(200, 'vision', 'HEAD_AWAY', 25))
        e.check_review()
        self.assertEqual((e.total_points, e.review_pending), (100, True))
        e.refresh(999999)
        e.check_review()
        self.assertEqual((e.total_points, e.review_pending), (100, True))

    def test_expired_combo_context_cannot_create_new_combo(self):
        e = RiskEngine(Config())
        e.add(RiskEvent(100, 'vision', 'HEAD_AWAY', 7))
        e.refresh(200)
        accepted, _ = e.add(RiskEvent(201, 'keyboard', 'CTRL+C', 2))
        self.assertEqual(len(accepted), 1)
        self.assertEqual(e.total_points, 7)
        self.assertEqual(e.score, 0)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_each_run_has_a_live_independent_csv(self):
        live = self.root / 'events.csv'
        first_id = uuid4().hex
        first = EventLogger(live, 100, first_id)
        first.write(RiskEvent(100, 'system', 'SESSION_START', 0, value=first_id), 0, 'NORMAL')
        first.write(RiskEvent(110, 'vision', 'HEAD_AWAY', 7), 7, 'NORMAL')
        evidence = first.session_path.read_bytes()
        self.assertEqual(evidence, live.read_bytes())
        second_id = uuid4().hex
        second = EventLogger(live, 200, second_id)
        second.write(RiskEvent(200, 'system', 'SESSION_START', 0, value=second_id), 0, 'NORMAL')
        self.assertNotEqual(first.session_path, second.session_path)
        self.assertEqual(first.session_path.read_bytes(), evidence)
        self.assertEqual(second.session_path.read_bytes(), live.read_bytes())
        self.assertEqual(len(list((self.root / 'sessions').glob('*.csv'))), 2)

    def test_custom_cbt_profile_persists_without_code_changes(self):
        profile = ExamProfile('cbt', '실험용 CBT', sys.executable, ('-c', 'pass'), str(self.root))
        path = self.root / 'profiles.json'
        save_profiles([profile], path)
        restored = load_profiles(path)[0]
        self.assertEqual(restored, profile)
        command, cwd = restored.command(self.root / 'session')
        self.assertEqual(command, [sys.executable, '-c', 'pass'])
        self.assertEqual(cwd, self.root)

    def test_invalid_profile_and_other_os_are_rejected(self):
        with self.assertRaises(ValueError):
            ExamProfile.from_dict({'id': 'x', 'name': 'x', 'executable': sys.executable, 'arguments': 'not a list'})
        profile = ExamProfile('x', 'x', sys.executable, working_directory=str(self.root), platform='linux' if os.name == 'nt' else 'windows')
        with self.assertRaises(ValueError): profile.command(self.root)

    def test_duplicate_instance_cannot_lock_same_log(self):
        with SessionLock(self.root / 'session.lock'):
            with self.assertRaises(RuntimeError):
                with SessionLock(self.root / 'session.lock'): pass
        with SessionLock(self.root / 'session.lock'): pass

    @unittest.skipUnless(os.name == 'nt', 'Windows foreground ownership')
    def test_title_keyword_only_applies_inside_owned_app(self):
        profile = ExamProfile('cbt', 'CBT', sys.executable, window_keywords=('exam',))
        target = ExamApplication(profile, self.root)
        target.process = MagicMock()
        api = MagicMock()
        api.GetWindowTextW.side_effect = lambda hwnd, buffer, size: setattr(buffer, 'value', 'CBT exam')
        with patch('exam_process.ctypes.WinDLL', return_value=api):
            target.process.owns_pid.return_value = True
            self.assertTrue(target.foreground_allowed())
            target.process.owns_pid.return_value = False
            self.assertFalse(target.foreground_allowed())
            target.process.owns_pid.return_value = True
            api.GetWindowTextW.side_effect = lambda hwnd, buffer, size: setattr(buffer, 'value', 'other tab')
            self.assertFalse(target.foreground_allowed())

    def run_monitor(self, weight):
        import main as app
        from typing_setup import TypingReference
        log = self.root / f'main-{weight}.csv'
        cfg = replace(Config(), log_path=log)
        profile = ExamProfile('demo', '시험 앱', sys.executable, working_directory=str(self.root))
        frames = iter([(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)])
        cap = MagicMock(); cap.isOpened.return_value = True; cap.get.return_value = 30
        cap.read.side_effect = lambda: next(frames)
        target = MagicMock(); target.start.return_value = 42; target.alive.return_value = True
        vision = MagicMock()
        vision.process.side_effect = lambda *args: VisionFrameResult(events=[RiskEvent(args[-1], 'vision', 'HEAD_AWAY', weight)])
        def review(path, *, control_log_path):
            self.assertEqual(control_log_path, log)
            target.stop.assert_not_called()
            target.close.assert_not_called()
            snapshot = read_snapshot(path)
            self.assertFalse(any(e.kind == 'SESSION_END' for e in snapshot.events))
            self.assertEqual(snapshot.cumulative, 100)
            return None
        with ExitStack() as stack:
            for name, value in [('Config', cfg), ('run_typing_setup', TypingReference(300)),
                                ('ExamApplication', target), ('VisionMonitor', vision),
                                ('KeyboardMonitor', MagicMock()),
                                ('WindowActivityMonitor', MagicMock(focus_lost=False, focus_lost_since=None)),
                                ('ClipboardChangeMonitor', MagicMock()), ('ExitMonitor', MagicMock())]:
                stack.enter_context(patch.object(app, name, return_value=value))
            stack.enter_context(patch.object(app.cv2, 'VideoCapture', return_value=cap))
            for name in ('namedWindow', 'imshow', 'destroyAllWindows'):
                stack.enter_context(patch.object(app.cv2, name))
            stack.enter_context(patch.object(app.cv2, 'waitKey', return_value=-1))
            stack.enter_context(patch.object(app, '_window_closed', return_value=False))
            for name in ('render', 'print_summary'):
                stack.enter_context(patch.object(app, name))
            stack.enter_context(patch.object(app.signal, 'signal'))
            opener = stack.enter_context(patch.object(app, '_open_reviewer', side_effect=review))
            app.main(profile=profile)
        return read_snapshot(log), opener

    def test_100_opens_review_while_app_is_alive_and_preserves_score(self):
        snapshot, opener = self.run_monitor(100)
        opener.assert_called_once()
        self.assertEqual(sum(e.kind == 'REVIEW_REQUESTED' for e in snapshot.events), 1)
        self.assertEqual(snapshot.cumulative, 100)
        self.assertEqual(snapshot.latest.kind, 'SESSION_END')

    def test_below_100_does_not_open_review(self):
        snapshot, opener = self.run_monitor(99)
        opener.assert_not_called()
        self.assertEqual(snapshot.cumulative, 99)

    @unittest.skipUnless(os.name == 'nt', 'Windows ownership test')
    def test_job_stops_owned_descendants_and_leaves_unrelated_process(self):
        outside = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], creationflags=subprocess.CREATE_NO_WINDOW)
        self.addCleanup(lambda: outside.poll() is None and outside.terminate())
        self.addCleanup(lambda: None)
        child_script = self.root / 'spawn.py'
        child_script.write_text('import subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])\n'
                                'open("child.pid","w").write(str(p.pid))\ntime.sleep(30)\n', encoding='utf-8')
        job = WindowsJob([sys.executable, str(child_script)], self.root)
        try:
            deadline = time.monotonic() + 5
            while not (self.root / 'child.pid').exists() and time.monotonic() < deadline: time.sleep(.05)
            child_pid = int((self.root / 'child.pid').read_text())
            self.assertTrue(job.owns_pid(child_pid))
            self.assertFalse(job.owns_pid(outside.pid))
            job.stop()
            self.assertFalse(job.alive())
            self.assertIsNone(outside.poll())
        finally:
            job.close(); outside.terminate(); outside.wait(timeout=5)


if __name__ == '__main__': unittest.main()
