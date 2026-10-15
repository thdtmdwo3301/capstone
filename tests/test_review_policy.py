"""Policy regression tests: synthetic events only, never the user's session log."""
from __future__ import annotations
import csv
import json
import queue
import sys
import tempfile
import threading
import unittest
from collections import deque
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from models import RiskEvent
from risk_engine import RiskEngine, OBSERVATIONS
from event_logger import EventLogger
from review_session import ReviewSession
from review_control import ReviewControl, request_control, command_status
from reviewer_data import read_snapshot, save_decision, LEGACY_FIELDS
from keyboard_monitor import KeyboardMonitor, keyboard
from activity_monitor import WindowActivityMonitor, allowed_external_window
from exam_profiles import ExamProfile, save_profiles, load_profiles
from main import _add_and_log


class LedgerPolicyTests(unittest.TestCase):
    def test_current_risk_hysteresis_does_not_control_review_pending(self):
        e = RiskEngine(Config(review_threshold=50))
        e.add(RiskEvent(100, 'vision', 'HEAD_AWAY', 50)); e.check_review()
        e.refresh(111.9)
        self.assertTrue(14 < e.score < 25)
        self.assertEqual(e.state, 'WATCH')
        self.assertTrue(e.review_pending)

    def test_waiting_48_then_3_calls_at_50(self):
        e = RiskEngine(Config(review_threshold=50))
        event = RiskEvent(100, 'vision', 'HEAD_AWAY', 48)
        e.add(event); self.assertFalse(e.check_review())
        e.refresh(109)
        self.assertAlmostEqual(e.score, 24)
        self.assertEqual(e.total_points, 48)
        e.refresh(121)
        self.assertEqual(e.score, 0); self.assertEqual(e.total_points, 48)
        e.add(RiskEvent(121, 'vision', 'GAZE_AWAY', 3))
        self.assertEqual((e.score, e.total_points), (3, 51))
        self.assertTrue(e.check_review()); self.assertTrue(e.review_pending)
        e.refresh(999); self.assertTrue(e.review_pending)
        self.assertEqual(e.total_points, 51)

    def test_event_id_prevents_duplicate_and_recalibration_reset(self):
        e = RiskEngine(Config())
        original = RiskEvent(100, 'vision', 'HEAD_AWAY', 7, confidence=.5)
        e.add(original); e.add(replace(original))
        e.set_active_visual(set()); e.set_sustained({}); e.refresh(999)
        self.assertEqual(e.total_points, 3.5)
        self.assertEqual(len(e.ledger), 1)
        self.assertEqual(e.score, 0)
        for t in range(1000, 1100):
            e.set_sustained({'HEAD_AWAY': 500}); e.refresh(t)
        self.assertEqual(e.total_points, 3.5)

    def test_observations_never_generate_direct_or_combo_points_both_modes(self):
        for mode in (False, True):
            e = RiskEngine(Config(test_mode=mode))
            e.add(RiskEvent(100, 'vision', 'HEAD_AWAY', 7))
            e.set_active_visual({'HEAD_AWAY'})
            for t in range(101, 111):
                for kind in OBSERVATIONS:
                    accepted, _ = e.add(RiskEvent(t, 'keyboard', kind, 99))
                    self.assertEqual(len(accepted), 1)
                    self.assertEqual(accepted[0].weight, 0)
            self.assertEqual(e.total_points, 7)
            self.assertEqual(e.event_counts['COMBINED_BEHAVIOR'], 0)

    def test_combo_has_actual_source_ids_and_pair_dedup_and_cooldown(self):
        for mode, weight in ((False, 10), (True, 12)):
            e = RiskEngine(Config(test_mode=mode))
            visual = RiskEvent(100, 'vision', 'HEAD_AWAY', 7)
            paste = RiskEvent(101, 'clipboard', 'LARGE_PASTE', 8)
            e.add(visual); events, _ = e.add(paste)
            self.assertEqual(events[-1].kind, 'COMBINED_BEHAVIOR')
            self.assertEqual(events[-1].weight, weight)
            self.assertEqual(set(events[-1].related_ids), {visual.event_id, paste.event_id})
            total = e.total_points
            self.assertFalse(e.add(replace(visual))[0]); self.assertFalse(e.add(replace(paste))[0])
            self.assertEqual(e.total_points, total)
            events, _ = e.add(RiskEvent(104.99, 'clipboard', 'LARGE_PASTE', 8))
            self.assertEqual(len(events), 1)
            events, _ = e.add(RiskEvent(105, 'clipboard', 'LARGE_PASTE', 8))
            self.assertEqual(len(events), 2)
            events, _ = e.add(RiskEvent(110.01, 'clipboard', 'LARGE_PASTE', 8))
            self.assertEqual(len(events), 1)

    def test_typing_only_zero_supported_max_two_no_fusion(self):
        for mode in (False, True):
            e = RiskEngine(Config(test_mode=mode))
            kinds = ('TYPING_SPEED_SPIKE','TYPING_RHYTHM_ANOMALY','LONG_IDLE_THEN_BURST')
            for kind in kinds:
                accepted, _ = e.add(RiskEvent(100, 'typing', kind, 10))
                self.assertEqual(accepted[0].weight, 0)
            e.add(RiskEvent(101, 'window', 'WINDOW_FOCUS_LOST', 8))
            for kind in kinds:
                accepted, _ = e.add(RiskEvent(102, 'typing', kind, 10))
                self.assertEqual(len(accepted), 1); self.assertEqual(accepted[0].weight, 2)
            self.assertEqual(e.total_points, 14)
            e.set_active_visual({'HEAD_AWAY'})
            accepted, _ = e.add(RiskEvent(120, 'typing', kinds[0], 10, confidence=.5))
            self.assertEqual(len(accepted), 1); self.assertEqual(e.total_points, 15)

    def test_required_weights_fail_fast_observation_weights_optional(self):
        with self.assertRaisesRegex(ValueError, '필수'):
            Config(event_weights={'HEAD_AWAY': 7})
        weights = {k:v for k,v in Config().event_weights.items() if k not in OBSERVATIONS}
        m = KeyboardMonitor(queue.Queue(), threading.Event(), Config(event_weights=weights))
        m._handle_shortcut('CTRL+C',100)
        self.assertEqual(m.output_queue.get_nowait().weight, 0)
        with self.assertRaises(ValueError):
            Config(review_threshold=float('nan'))


class InputPolicyTests(unittest.TestCase):
    def test_removed_keys_and_modified_escape_have_no_events_or_errors(self):
        for mode in (False, True):
            for modifiers, key in [
                ([keyboard.Key.ctrl_l], c) for c in 'tnlw'
            ] + [([keyboard.Key.ctrl_l,keyboard.Key.shift], c) for c in 'ti'] + [
                ([keyboard.Key.ctrl_l,keyboard.Key.shift], keyboard.Key.esc),
                ([keyboard.Key.alt_l], keyboard.Key.f4), ([], keyboard.Key.f12)]:
                q, stop = queue.Queue(), threading.Event()
                m = KeyboardMonitor(q, stop, Config(test_mode=mode))
                k = keyboard.KeyCode.from_char(key) if isinstance(key, str) else key
                for item in modifiers+[k]: m._on_press(item)
                for item in modifiers+[k]: m._on_release(item)
                self.assertTrue(q.empty()); self.assertFalse(stop.is_set())
            self.assertFalse({'CTRL+T','CTRL+N','CTRL+L','CTRL+W','CTRL+SHIFT+T',
                              'CTRL+SHIFT+I','CTRL+SHIFT+ESC','ALT+F4','F12'} & Config(test_mode=mode).event_weights.keys())

    def test_paste_boundaries_and_single_input_contribution(self):
        for mode in (False, True):
            for size in (0,79,80,199,200,499,500,1000):
                for shortcut in ('CTRL+V','SHIFT+INSERT'):
                    with self.subTest(mode=mode, size=size, shortcut=shortcut):
                        q=queue.Queue(); cfg=Config(test_mode=mode)
                        m=KeyboardMonitor(q,threading.Event(),cfg)
                        with patch('keyboard_monitor.clipboard_metadata',return_value={'type':'TEXT','chars':size,'lines':3}):
                            m._handle_shortcut(shortcut,100)
                        events=list(q.queue)
                        self.assertEqual(events[0].weight,0)
                        if size<80: self.assertEqual(len(events),1); continue
                        expected=15 if mode else 8 if size<200 else 12 if size<500 else 18
                        self.assertEqual(events[1].weight,expected)
                        self.assertIn('삽입 성공·외부 출처 미확인',events[1].context)
                        e=RiskEngine(cfg)
                        for item in events: e.add(item); e.add(replace(item))
                        self.assertAlmostEqual(e.total_points, expected*min(1,.5+size/400))
                        self.assertNotIn('COMBINED_BEHAVIOR', e.event_counts)

    def test_configured_base_scales_paste_weights(self):
        weights={**Config().event_weights, 'LARGE_PASTE': 10}
        cfg=Config(event_weights=weights)
        self.assertEqual([cfg.large_paste_weight(n) for n in (79,80,199,200,499,500)], [0,10,10,15,15,22.5])

    def test_normal_self_code_reuse_is_still_scored_limitation(self):
        # The monitor cannot distinguish authorized self-reuse from outside code.
        cfg=Config()
        m=KeyboardMonitor(queue.Queue(),threading.Event(),cfg)
        with patch('keyboard_monitor.clipboard_metadata',return_value={'type':'TEXT','chars':200,'lines':6}):
            m._handle_shortcut('CTRL+V',100)
        e=RiskEngine(cfg)
        for item in list(m.output_queue.queue): e.add(item)
        self.assertEqual(e.total_points,12)

    def test_proctor_input_not_exam_typing_paste_or_escape(self):
        m=KeyboardMonitor(queue.Queue(),threading.Event(),Config())
        m.operator_active=lambda:True
        for key in [keyboard.Key.ctrl_l,keyboard.KeyCode.from_char('v'),keyboard.Key.esc]:
            m._on_press(key); m._on_release(key)
        self.assertTrue(m.output_queue.empty())
        self.assertFalse(m.stop_event.is_set())
        self.assertEqual(m.total_typing_keys,0)
        self.assertFalse(m.pressed)


class FocusPolicyTests(unittest.TestCase):
    def test_allowed_switches_and_mouse_only_unallowed_exit(self):
        for mode, weight in ((False,8),(True,14)):
            q=queue.Queue(); m=WindowActivityMonitor(q,threading.Event(),Config(test_mode=mode))
            for t,name in [(100,'Code.exe'),(101,'browser.exe'),(104,'terminal.exe')]:
                m.observe(t,name,True)
            self.assertTrue(q.empty())
            m.observe(105,'other.exe',False); m.observe(106.99,'other.exe',False)
            self.assertTrue(q.empty())
            m.observe(107,'other.exe',False)
            self.assertEqual(q.get_nowait().weight,weight)
            m.observe(116.99,'other.exe',False); self.assertTrue(q.empty())
            m.observe(117,'other.exe',False); self.assertEqual(q.get_nowait().weight,weight)
            m.observe(118,'Code.exe',True)
            self.assertFalse(m.focus_lost); self.assertEqual(q.get_nowait().weight,0)

    def test_profile_external_rule_requires_path_and_specific_title(self):
        executable=str(Path(sys.executable).resolve())
        rules=({'executable':executable,'title_keywords':['Approved problem']},)
        self.assertTrue(allowed_external_window(rules,'Approved problem 1',executable))
        self.assertFalse(allowed_external_window(rules,'Other page',executable))
        self.assertFalse(allowed_external_window(rules,'Approved problem 1',executable+'.other'))
        with tempfile.TemporaryDirectory() as tmp:
            p=ExamProfile('cbt','CBT',executable,allowed_windows=rules)
            dest=Path(tmp)/'profiles.json'; save_profiles([p],dest)
            self.assertEqual(load_profiles(dest)[0].allowed_windows,rules)
        with self.assertRaises(ValueError):
            ExamProfile.from_dict({'id':'x','name':'x','executable':executable,
                                   'allowed_windows':[{'executable':'Code.exe','title_keywords':[]}]})

    def test_proctor_focus_time_does_not_become_window_loss_duration(self):
        q=queue.Queue(); m=WindowActivityMonitor(q,threading.Event(),Config())
        m.observe(100,'Code.exe',True); m.observe(101,'other.exe',False)
        m.observe(102,'reviewer',False,True); m.observe(200,'reviewer',False,True)
        m.observe(201,'other.exe',False)
        self.assertTrue(q.empty())
        m.observe(203,'other.exe',False)
        event=q.get_nowait()
        self.assertEqual(event.duration,2)


class LiveReviewPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.sid=uuid4().hex
        self.cfg=Config(log_path=self.root/'events.csv',review_threshold=50)
        self.engine=RiskEngine(self.cfg)
        self.logger=EventLogger(self.cfg.log_path,100,self.sid); self.logger.engine=self.engine
        self.logger.write(RiskEvent(100,'system','SESSION_START',0,value=self.sid),0,'NORMAL')
        self.controller=ReviewControl(self.cfg.log_path,self.sid)
        self.process=MagicMock(pid=123); self.process.poll.return_value=None
        self.opener=MagicMock(return_value=self.process)
        self.review=ReviewSession(self.engine,self.logger,self.controller,self.opener)

    def add(self,points,at=100):
        event=RiskEvent(at,'vision','HEAD_AWAY',points)
        _add_and_log(self.engine,self.logger,event,deque())
        return event

    def tick(self,now):
        with patch('review_session.time.monotonic',return_value=now):
            self.engine.refresh(now); self.review.tick(now)

    def decide(self,snapshot,decision='계속 응시'):
        saved=save_decision(snapshot,self.root/'reviews',reviewer='synthetic proctor',
                            decision=decision,reason='synthetic review only')
        commandpath=request_control(self.cfg.log_path,saved)
        self.controller.last_poll=-1
        command=self.controller.poll()
        self.assertIsNotNone(command)
        self.review.apply(command)
        return saved,commandpath

    def test_only_one_async_window_close_does_not_acknowledge(self):
        self.add(51); self.tick(100)
        for now in range(101,130): self.tick(now)
        self.opener.assert_called_once()
        self.process.poll.return_value=0
        for now in range(130,140): self.tick(now)
        self.opener.assert_called_once()
        self.assertTrue(self.engine.review_pending); self.assertEqual(self.engine.reviewed_points,0)
        self.assertEqual(sum(e.kind=='REVIEW_REQUESTED' for e in read_snapshot(self.cfg.log_path).events),1)

    def test_continue_preserves_c_and_events_arriving_during_review_realert(self):
        self.add(51); self.tick(100)
        cutoff=read_snapshot(self.cfg.log_path)
        self.add(55,110); self.tick(110)
        saved,path=self.decide(cutoff)
        self.assertEqual(command_status(path)[0],'completed')
        self.assertEqual((self.engine.total_points,self.engine.reviewed_points,self.engine.unreviewed_points),(106,51,55))
        self.tick(111)
        self.assertTrue(self.engine.review_pending); self.assertEqual(self.engine.review_generation,2)
        self.opener.assert_called_once()  # existing live reviewer updates, no extra window
        record=json.loads(saved.read_text(encoding='utf-8'))
        self.assertEqual(record['reviewed_cumulative'],51)
        evidence=read_snapshot(saved.with_name('reviewed_events.csv'))
        self.assertEqual(evidence.cumulative,51)
        live=read_snapshot(self.cfg.log_path)
        self.assertEqual(live.cumulative,106); self.assertEqual(live.latest.reviewed,51)
        self.assertFalse(live.warning)

    def test_defer_preserves_pending_and_no_watermark_change(self):
        self.add(51); self.tick(100)
        self.decide(read_snapshot(self.cfg.log_path),'판단 보류')
        self.assertTrue(self.engine.review_pending); self.assertEqual(self.engine.reviewed_points,0)

    def test_duplicate_csv_delivery_and_more_than_100_are_valid(self):
        event=self.add(120)
        _add_and_log(self.engine,self.logger,replace(event),deque())
        self.logger.write(event,self.engine.score,self.engine.state)
        self.tick(100)
        snapshot=read_snapshot(self.cfg.log_path)
        self.assertEqual(snapshot.cumulative,120)
        self.assertFalse(snapshot.warning)
        self.assertEqual(sum(e.row['EventId']==event.event_id for e in snapshot.events),1)
        self.decide(snapshot)
        self.assertEqual(self.engine.reviewed_points,120)

    def test_wrong_session_and_forged_cutoff_are_rejected(self):
        self.add(51); self.tick(100)
        saved=save_decision(read_snapshot(self.cfg.log_path),self.root/'reviews',
                            reviewer='test',decision='계속 응시',reason='synthetic')
        path=request_control(self.cfg.log_path,saved)
        self.assertIsNone(ReviewControl(self.cfg.log_path,uuid4().hex).poll())
        record=json.loads(saved.read_text(encoding='utf-8')); record['reviewed_cumulative']=999
        saved.write_text(json.dumps(record),encoding='utf-8')
        self.assertIsNone(self.controller.poll()); self.assertEqual(command_status(path)[0],'rejected')
        self.assertEqual(self.engine.reviewed_points,0)

    def test_retest_archive_new_id_zero_without_overwrite(self):
        self.add(130); self.tick(100)
        old=self.logger.session_path.read_bytes()
        saved,path=self.decide(read_snapshot(self.cfg.log_path),'재시험 요청')
        self.assertEqual(command_status(path)[0],'received')  # no completion before managed cleanup
        self.assertTrue(saved.exists())
        sid2=uuid4().hex
        second=EventLogger(self.cfg.log_path,200,sid2); second.engine=RiskEngine(self.cfg)
        second.write(RiskEvent(200,'system','SESSION_START',0,value=sid2),0,'NORMAL')
        self.assertEqual(self.logger.session_path.read_bytes()[:len(old)],old)
        self.assertNotEqual(self.logger.session_path,second.session_path)
        self.assertEqual(read_snapshot(second.path).cumulative,0)
        with self.assertRaises(ValueError): request_control(second.path,saved)

    def test_legacy_csv_remains_readable(self):
        path=self.root/'legacy.csv'
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f);writer.writerow(LEGACY_FIELDS)
            writer.writerow(['then','0','system','SESSION_START','','0','1','0','0','NORMAL',''])
            writer.writerow(['then','3','vision','HEAD_AWAY','LEFT','2','1','7','7','NORMAL','old'])
        snapshot=read_snapshot(path)
        self.assertEqual(len(snapshot.events),2)
        self.assertNotIn('CumulativeScore',snapshot.latest.row)

    def test_modern_log_identity_without_start_value_uses_explicit_session_id(self):
        logger=EventLogger(self.root/'standalone.csv',100)
        logger.write(RiskEvent(100,'system','SESSION_START',0),0,'NORMAL')
        logger.write(RiskEvent(101,'vision','HEAD_AWAY',7),7,'NORMAL')
        snapshot=read_snapshot(logger.path)
        self.assertEqual(snapshot.session_id,logger.session_id)
        self.assertEqual(snapshot.cumulative,7)

    def test_tk_review_freezes_cutoff_refresh_does_not_complete_and_graph_handles_120(self):
        import tkinter as tk
        from reviewer import ReviewerApp
        self.add(51); self.tick(100)
        root=tk.Tk(); root.withdraw()
        try:
            app=ReviewerApp(root,self.cfg.log_path)
            cut=app.review_snapshot
            self.add(69,110); self.tick(110)
            app._refresh(True); root.update_idletasks()
            self.assertEqual(app.snapshot.cumulative,120)
            self.assertIs(app.review_snapshot,cut)
            self.assertEqual(self.engine.reviewed_points,0)
            self.assertIn('69.00',app.scope_notice.get())
            self.assertTrue(app.chart.find_all())
            for item in app.chart.find_all():
                if app.chart.type(item)=='line':
                    ys=app.chart.coords(item)[1::2]
                    self.assertTrue(all(0<=y<=110 for y in ys))
            app._include_latest()
            self.assertEqual(app.review_snapshot.cumulative,120)
            app.close()
            self.assertTrue(self.engine.review_pending)
        finally:
            if not app.closed: root.destroy()


if __name__=='__main__': unittest.main()

