"""Real owned Windows process + real CSV/decisions, simulated camera/hook sources."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as app
from config import Config
from exam_profiles import ExamProfile
from exam_process import ExamApplication
from models import RiskEvent, VisionFrameResult
from review_control import ReviewControl, request_control
from reviewer_data import read_snapshot, save_decision
from typing_setup import TypingReference


@unittest.skipUnless(os.name == 'nt', 'real Windows Job ownership')
class LivePolicyIntegrationTests(unittest.TestCase):
    def exercise(self, restart):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=replace(Config(),log_path=root/'events.csv',review_threshold=50)
            profile=ExamProfile('synthetic','synthetic CBT',sys.executable,
                                ('-c','import time; time.sleep(60)'),str(root))
            outside=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            targets=[]; controllers=[]; sessions=[]; captures=[]; decisions=[]
            state={'run':-1,'frame':0}
            def make_target(*args):
                target=ExamApplication(*args); targets.append(target); return target
            def make_control(*args):
                controller=ReviewControl(*args); controllers.append(controller); return controller
            def decide(snapshot, value):
                saved=save_decision(snapshot,root/'reviews',reviewer='integration tester',
                                    decision=value,reason='synthetic evidence only')
                decisions.append(saved)
                request_control(cfg.log_path,saved); controllers[-1].last_poll=-1
            def make_capture(*args):
                state['run']+=1; state['frame']=0
                cap=MagicMock();cap.isOpened.return_value=True;cap.get.return_value=30
                def read():
                    state['frame']+=1
                    count=state['frame']; run=state['run']
                    self.assertTrue(targets[-1].alive())
                    self.assertIsNone(outside.poll())
                    snapshot=read_snapshot(cfg.log_path)
                    if count==1:
                        sessions.append(snapshot.session_id)
                        self.assertEqual(snapshot.cumulative,0)
                        if run==1: decide(snapshot,'시험 중단 결정')
                    if run==0 and count==2:
                        self.assertEqual(snapshot.cumulative,51)
                        self.assertEqual(snapshot.latest.row['ReviewPending'],'1')
                        decide(snapshot,'계속 응시')
                    if run==0 and count==3:
                        self.assertEqual(snapshot.cumulative,106)
                        self.assertEqual(snapshot.latest.reviewed,51)
                    if run==0 and count==4:
                        self.assertEqual(snapshot.latest.row['ReviewPending'],'1')
                        self.assertEqual(sum(e.kind=='REVIEW_REQUESTED' for e in snapshot.events),2)
                        decide(snapshot,'재시험 요청' if restart else '퇴장 조치 결정')
                    if count>5: self.fail('terminal decision was not consumed by the live loop')
                    return True,np.zeros((480,640,3),dtype=np.uint8)
                cap.read.side_effect=read;captures.append(cap);return cap
            vision=MagicMock()
            vision.process.side_effect=lambda *args:VisionFrameResult(events=[RiskEvent(
                args[-1],'vision','HEAD_AWAY',51 if state['run']==0 and state['frame']==1 else 55 if state['run']==0 and state['frame']==2 else 0)])
            popup=MagicMock(pid=-1);popup.poll.return_value=None
            try:
                with ExitStack() as stack:
                    for name,value in [('Config',cfg),('run_typing_setup',TypingReference(300)),
                                       ('VisionMonitor',vision),('KeyboardMonitor',MagicMock()),
                                       ('WindowActivityMonitor',MagicMock(focus_lost=False,focus_lost_since=None)),
                                       ('ClipboardChangeMonitor',MagicMock()),('ExitMonitor',MagicMock())]:
                        stack.enter_context(patch.object(app,name,return_value=value))
                    stack.enter_context(patch.object(app,'ExamApplication',side_effect=make_target))
                    stack.enter_context(patch.object(app,'ReviewControl',side_effect=make_control))
                    stack.enter_context(patch.object(app.cv2,'VideoCapture',side_effect=make_capture))
                    for name in ('namedWindow','imshow','destroyAllWindows'):
                        stack.enter_context(patch.object(app.cv2,name))
                    stack.enter_context(patch.object(app.cv2,'waitKey',return_value=-1))
                    stack.enter_context(patch.object(app,'_window_closed',return_value=False))
                    opener=stack.enter_context(patch.object(app,'_open_reviewer',return_value=popup))
                    for name in ('render','print_summary'):
                        stack.enter_context(patch.object(app,name))
                    stack.enter_context(patch.object(app.signal,'signal'))
                    app.run_application(profile)
                self.assertEqual(len(targets),2 if restart else 1)
                self.assertEqual(len(set(sessions)),len(targets))
                self.assertTrue(all(not target.alive() for target in targets))
                self.assertIsNone(outside.poll())
                self.assertEqual(opener.call_count,1)
                for path in decisions:
                    result=json.loads(path.with_name('control_result.json').read_text(encoding='utf-8'))
                    self.assertEqual(result['status'],'completed')
                    self.assertEqual(result['session_id'],json.loads(path.read_text(encoding='utf-8'))['session_id'])
                archives=[read_snapshot(path) for path in (root/'sessions').glob('*.csv')]
                self.assertEqual(len(archives),len(targets))
                old=next(s for s in archives if s.session_id==sessions[0])
                self.assertEqual(old.cumulative,106)
                self.assertEqual(old.latest.reviewed,51)
                self.assertEqual(old.latest.kind,'SESSION_END')
                self.assertEqual(captures[0].read.call_count,4)
                for cap in captures: cap.release.assert_called_once()
            finally:
                for target in targets: target.close()
                outside.terminate(); outside.wait(timeout=5)

    def test_threshold_continue_realert_then_human_expel_only_stops_owned_app(self):
        self.exercise(False)

    def test_human_retest_starts_new_session_preserves_old_evidence_and_unrelated_app(self):
        self.exercise(True)


if __name__=='__main__':unittest.main()
