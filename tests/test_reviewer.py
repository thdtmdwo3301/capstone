from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from event_logger import EventLogger
from models import RiskEvent
from reviewer_data import FIELDS, read_snapshot, save_decision, decision_history
from review_control import ReviewControl, request_control, command_status


class ProctorReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'events.csv'
        self.session_id = uuid4().hex
        self.logger = EventLogger(self.path, 100, self.session_id)
        self.logger.write(RiskEvent(100, 'system', 'SESSION_START', 0, value=self.session_id), 0, 'NORMAL')
        self.logger.write(RiskEvent(102, 'vision', 'HEAD_AWAY', 7, value='RIGHT', duration=2,
                                   confidence=.5, context='yaw=-30,\n설명 "검토"'), 3.5, 'NORMAL')

    def decision(self, value='시험 중단 결정', snapshot=None):
        return save_decision(snapshot or read_snapshot(self.path), self.root / 'reviews',
                             reviewer='테스트 감독관', decision=value, reason='테스트 기록', selected_indices=(2,))

    def test_korean_multiline_csv_and_stable_session_on_append(self):
        before = read_snapshot(self.path)
        self.assertEqual(before.session_id, self.session_id)
        self.assertEqual(before.events[1].row['Context'], 'yaw=-30,\n설명 "검토"')
        self.logger.write(RiskEvent(103, 'vision', 'GAZE_AWAY', 6), 9, 'NORMAL')
        after = read_snapshot(self.path)
        self.assertEqual(before.session_id, after.session_id)
        self.assertEqual(after.counts['HEAD_AWAY'], 1)
        self.assertEqual(after.peak, 9)

    def test_partial_final_row_is_not_used_as_evidence(self):
        with self.path.open('ab') as stream:
            stream.write(b'2026-10-14T13:00:00,3,vision,"HEAD')
        snapshot = read_snapshot(self.path)
        self.assertTrue(snapshot.pending)
        self.assertEqual(len(snapshot.events), 2)

    def test_invalid_numeric_values_are_not_scored(self):
        with self.path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['now', '3', 'vision', 'HEAD_AWAY', 'RIGHT', '2', '1', 'nan', '100', 'SUSPICIOUS', ''])
        snapshot = read_snapshot(self.path)
        self.assertEqual(len(snapshot.events), 2)
        self.assertIn('1개', snapshot.warning)

    def test_wrong_schema_and_multiple_sessions_are_rejected(self):
        other = self.root / 'other.csv'
        other.write_text('name,value\nx,1\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            read_snapshot(other)
        self.logger.write(RiskEvent(104, 'system', 'SESSION_START', 0, value=uuid4().hex), 0, 'NORMAL')
        with self.assertRaises(ValueError):
            read_snapshot(self.path)

    def test_decision_copies_reviewed_evidence_and_preserves_source(self):
        original = self.path.read_bytes()
        target = self.decision()
        record = json.loads(target.read_text(encoding='utf-8'))
        self.assertEqual(record['selected_event_indices'], [2])
        self.assertEqual(record['decision'], '시험 중단 결정')
        evidence = target.with_name('reviewed_events.csv').read_bytes()
        self.assertEqual(record['evidence_sha256'], hashlib.sha256(evidence).hexdigest())
        self.assertEqual(read_snapshot(target.with_name('reviewed_events.csv')).session_id, self.session_id)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(len(decision_history(self.root / 'reviews', self.session_id)), 1)

    def test_live_append_does_not_claim_unreviewed_events(self):
        snapshot = read_snapshot(self.path)
        self.logger.write(RiskEvent(104, 'vision', 'FACE_MISSING', 10), 13, 'NORMAL')
        target = self.decision(snapshot=snapshot)
        record = json.loads(target.read_text(encoding='utf-8'))
        self.assertEqual(record['reviewed_event_count'], 2)
        self.assertEqual(len(read_snapshot(target.with_name('reviewed_events.csv')).events), 2)

    def test_overwritten_session_cannot_receive_old_decision(self):
        snapshot = read_snapshot(self.path)
        new_id = uuid4().hex
        new_logger = EventLogger(self.path, 200, new_id)
        new_logger.write(RiskEvent(200, 'system', 'SESSION_START', 0, value=new_id), 0, 'NORMAL')
        with self.assertRaises(ValueError):
            self.decision(snapshot=snapshot)
        self.assertEqual(read_snapshot(new_logger.archived_path).session_id, snapshot.session_id)

    def test_human_reason_and_valid_event_selection_required(self):
        snapshot = read_snapshot(self.path)
        for reviewer, reason, selected in (('', '근거', (2,)), ('감독관', '', (2,)), ('감독관', '근거', (999,))):
            with self.subTest(reviewer=reviewer, reason=reason, selected=selected):
                with self.assertRaises(ValueError):
                    save_decision(snapshot, self.root / 'reviews', reviewer=reviewer,
                                  reason=reason, decision='판단 보류', selected_indices=selected)

    def test_stop_and_restart_commands_match_saved_human_decision(self):
        for decision, expected in (('시험 중단 결정', 'stop'), ('재시험 요청', 'restart')):
            with self.subTest(decision=decision):
                target = self.decision(decision)
                path = request_control(self.path, target)
                self.assertEqual(command_status(path)[0], 'pending')
                control = ReviewControl(self.path, self.session_id)
                command = control.poll()
                self.assertEqual(command.action, expected)
                self.assertEqual(command_status(path)[0], 'received')
                control.complete(command, message='완료')
                self.assertEqual(command_status(path), ('completed', '완료'))
                control.last_poll = -1
                self.assertIsNone(control.poll())
                with self.assertRaises(ValueError):
                    request_control(self.path, target)

    def test_continue_sends_explicit_command_and_wrong_session_is_ignored(self):
        self.assertIsNotNone(request_control(self.path, self.decision('계속 응시')))
        path = request_control(self.path, self.decision())
        self.assertIsNone(ReviewControl(self.path, uuid4().hex).poll())
        self.assertEqual(command_status(path)[0], 'pending')

    def test_forged_action_is_rejected(self):
        path = request_control(self.path, self.decision('시험 중단 결정'))
        data = json.loads(path.read_text(encoding='utf-8'))
        data['action'] = 'restart'
        path.write_text(json.dumps(data), encoding='utf-8')
        self.assertIsNone(ReviewControl(self.path, self.session_id).poll())
        self.assertEqual(command_status(path)[0], 'rejected')

    def test_ended_session_cannot_receive_control(self):
        target = self.decision()
        self.logger.write(RiskEvent(110, 'system', 'SESSION_END', 0), 1, 'NORMAL')
        with self.assertRaises(ValueError):
            request_control(self.path, target)

    def test_application_restarts_once_without_duplicate_reviewer(self):
        import main as app
        controller = MagicMock()
        command = SimpleNamespace(action='restart')
        with patch.object(app, 'main', side_effect=[(controller, command), None]) as run:
            app.run_application(profile='test-profile')
        self.assertEqual([call.kwargs['profile'] for call in run.call_args_list], ['test-profile', 'test-profile'])
        controller.complete.assert_called_once()

    def test_application_stop_does_not_restart(self):
        import main as app
        controller = MagicMock()
        command = SimpleNamespace(action='stop')
        with patch.object(app, 'main', return_value=(controller, command)) as run:
            app.run_application(profile='test-profile')
        self.assertEqual(run.call_count, 1)
        controller.complete.assert_called_once()

    def test_main_consumes_control_releases_camera_and_logs_end(self):
        import main as app
        from typing_setup import TypingReference
        cfg = replace(Config(), log_path=self.root / 'main-test.csv')
        controller = MagicMock()
        command = SimpleNamespace(action='stop', decision='시험 중단 결정', review_id='test-review', reviewer='테스트 감독관', reason='synthetic', reviewed_sequence=0, reviewed_cumulative=0)
        controller.poll.return_value = command
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30
        with patch.object(app, 'Config', return_value=cfg), \
             patch.object(app, 'run_typing_setup', return_value=TypingReference(300)), \
             patch.object(app, 'ReviewControl', return_value=controller), \
             patch.object(app, 'VisionMonitor'), patch.object(app, 'KeyboardMonitor'), \
             patch.object(app, 'WindowActivityMonitor'), patch.object(app, 'ClipboardChangeMonitor'), \
             patch.object(app, 'ExitMonitor'), patch.object(app.cv2, 'VideoCapture', return_value=cap), \
             patch.object(app.cv2, 'namedWindow'), patch.object(app.cv2, 'destroyAllWindows'), \
             patch.object(app, 'print_summary'), patch.object(app.signal, 'signal'):
            result = app.main()
        cap.release.assert_called_once()
        cap.read.assert_not_called()
        self.assertEqual(result, (controller, command))
        snapshot = read_snapshot(cfg.log_path)
        self.assertEqual(len(snapshot.session_id), 32)
        self.assertEqual(snapshot.events[-1].kind, 'SESSION_END')
        self.assertEqual(snapshot.events[-1].row['Value'], 'STOP')
        self.assertIn('PROCTOR_DECISION', [event.kind for event in snapshot.events])


if __name__ == '__main__':
    unittest.main()
