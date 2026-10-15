"""Regression: auto reviewer reads sessions/*.csv but controls the live monitor root."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import Config
from event_logger import EventLogger
from main import _open_reviewer
from models import RiskEvent
from risk_engine import RiskEngine
from review_control import ReviewControl, command_status
from review_session import ReviewSession
from reviewer import ReviewerApp, control_log_path_for
from reviewer_data import read_snapshot


class AutoReviewerPathTests(unittest.TestCase):
    def test_auto_launch_archive_save_continue_reaches_live_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            live=directory/'custom-live-name.csv'
            cfg=Config(log_path=live, review_threshold=50)
            engine=RiskEngine(cfg);sid=uuid4().hex
            logger=EventLogger(live,100,sid);logger.engine=engine
            logger.write(RiskEvent(100,'system','SESSION_START',0,value=sid),0,'NORMAL')
            def add(points,now):
                for event in engine.add(RiskEvent(now,'vision','HEAD_AWAY',points))[0]:
                    logger.write(event,engine.score,engine.state)
            add(51,100)
            controller=ReviewControl(live,sid)
            review=ReviewSession(engine,logger,controller,_open_reviewer)
            process=MagicMock(pid=-1); process.poll.return_value=None
            with patch('main.subprocess.Popen',return_value=process) as launch:
                review.tick(100)
            args=launch.call_args.args[0][2:]
            parser=argparse.ArgumentParser()
            parser.add_argument('--log',type=Path)
            parser.add_argument('--control-log',type=Path)
            parser.add_argument('--auto-review',action='store_true')
            cli=parser.parse_args(args)
            self.assertEqual(cli.log,logger.session_path)
            self.assertEqual(cli.control_log,live)
            root=tk.Tk();root.withdraw()
            ui=ReviewerApp(root,cli.log,cli.auto_review,cli.control_log)
            try:
                self.assertEqual(ui.path,logger.session_path)
                self.assertEqual(ui.review_dir,directory/'reviews')
                add(55,110);ui._refresh(True)
                self.assertEqual(ui.review_snapshot.cumulative,51)
                ui.reviewer.set('synthetic proctor');ui.decision.set('계속 응시')
                ui.reason.insert('1.0','synthetic authorized behavior')
                with patch('reviewer.messagebox.showerror') as error:
                    ui._save()
                error.assert_not_called()
                self.assertIsNotNone(ui.pending_command)
                self.assertEqual(ui.pending_command.parent,directory/'review_commands'/sid)
                self.assertFalse((directory/'sessions'/'review_commands').exists())
                command=controller.poll()
                self.assertIsNotNone(command)
                review.apply(command)
                self.assertEqual(command_status(ui.pending_command)[0],'completed')
                self.assertEqual(engine.reviewed_points,51)
                self.assertEqual(engine.total_points,106)
                self.assertEqual(engine.unreviewed_points,55)
                self.assertEqual(read_snapshot(logger.session_path).latest.reviewed,51)
                ui._poll()
                self.assertIsNone(ui.pending_command)
                self.assertEqual(ui.snapshot.latest.reviewed,51)
            finally:
                ui.close()

    def test_old_running_monitor_archive_only_launch_resolves_configured_live_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            live=Path(tmp)/'configured.csv'
            archive=Path(tmp)/'sessions'/'session_timestamp.csv'
            with patch('reviewer.Config',return_value=Config(log_path=live)):
                self.assertEqual(control_log_path_for(archive),live.resolve())
                root=tk.Tk();root.withdraw()
                ui=ReviewerApp(root,archive,True)
                try:
                    self.assertEqual(ui.review_dir,live.parent/'reviews')
                    self.assertEqual(ui.live_path,live)
                finally:
                    ui.close()

    def test_previous_archive_cannot_send_control_to_new_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            from review_control import request_control
            from reviewer_data import save_decision
            root=Path(tmp);live=root/'live.csv'
            sid=uuid4().hex;first=EventLogger(live,100,sid)
            first.write(RiskEvent(100,'system','SESSION_START',0,value=sid),0,'NORMAL')
            first.write(RiskEvent(101,'vision','HEAD_AWAY',51),51,'SUSPICIOUS')
            snapshot=read_snapshot(first.session_path)
            saved=save_decision(snapshot,root/'reviews',reviewer='test',decision='계속 응시',reason='synthetic')
            new_id=uuid4().hex;second=EventLogger(live,200,new_id)
            second.write(RiskEvent(200,'system','SESSION_START',0,value=new_id),0,'NORMAL')
            with self.assertRaises(ValueError):
                request_control(live,saved)
            self.assertFalse((root/'review_commands').exists())


if __name__=='__main__': unittest.main()

