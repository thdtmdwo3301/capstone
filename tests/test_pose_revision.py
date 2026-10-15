import sys, unittest, math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_monitor import _estimate_head_pose, SustainedEventTracker
from vision_monitor import VisionMonitor, Calibrator, MeasurementSmoother
from config import Config
from models import GazeMeasurement, RiskEvent
from risk_engine import RiskEngine
from keyboard_monitor import KeyboardMonitor, keyboard
from event_logger import EventLogger
import queue, threading, tempfile, csv

class PoseRevisionTests(unittest.TestCase):
    def landmarks(self, yaw=0, z=2400):
        obj=np.array([[0,0,0],[0,330,-65],[-225,-170,-135],[225,-170,-135],[-150,150,-125],[150,150,-125]],dtype=float)
        r=cv2.Rodrigues(np.array([[math.cos(math.radians(yaw)),0,math.sin(math.radians(yaw))],[0,1,0],[-math.sin(math.radians(yaw)),0,math.cos(math.radians(yaw))]],dtype=float))[0]
        f=640/(2*math.tan(math.radians(60)/2));k=np.array([[f,0,320],[0,f,240],[0,0,1]],dtype=float)
        pts=cv2.projectPoints(obj,r,np.array([[0.],[0.],[float(z)]]),k,np.zeros((4,1)))[0].reshape(-1,2)
        lm=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
        for i,p in zip([1,152,33,263,61,291],pts):lm[i]=SimpleNamespace(x=p[0]/640,y=p[1]/480)
        return lm
    def test_front_and_side_at_two_scales(self):
        for z in [1800,4000]:
            for yaw in [-55,0,55]:
                with self.subTest(z=z,yaw=yaw):
                    pose=_estimate_head_pose(self.landmarks(yaw,z),640,480)
                    self.assertIsNotNone(pose)
                    self.assertAlmostEqual(pose[0],yaw,places=3)
                    self.assertAlmostEqual(pose[1],0,places=3)
    def test_behind_camera_rejected(self):
        with patch('vision_monitor.cv2.solvePnP',return_value=(True,np.zeros((3,1)),np.array([[0.],[0.],[-2400.]]))):
            self.assertIsNone(_estimate_head_pose(self.landmarks(),640,480))
    def test_bad_projection_rejected(self):
        with patch('vision_monitor.cv2.solvePnP',return_value=(True,np.zeros((3,1)),np.array([[5000.],[0.],[2400.]]))):
            self.assertIsNone(_estimate_head_pose(self.landmarks(),640,480))
    def tick(self,tr,t,active):
        return tr.update(kind='HEAD_AWAY',active=active,now=t,min_duration=2,repeat_sec=8,weight=7,value='LEFT',confidence=1,context='',release_grace_sec=.5)
    def test_brief_gap_keeps_confirmed_event(self):
        tr=SustainedEventTracker();self.tick(tr,0,True);self.tick(tr,2,True)
        self.assertFalse(self.tick(tr,3,False));self.assertFalse(self.tick(tr,3.2,True))
        self.assertEqual(tr.states['HEAD_AWAY']['started'],0)
        self.assertTrue(tr.states['HEAD_AWAY']['emitted'])
    def test_stable_normal_ends_event(self):
        tr=SustainedEventTracker();self.tick(tr,0,True);self.tick(tr,2,True)
        self.tick(tr,3,False);events=self.tick(tr,3.6,False)
        self.assertEqual(events[0].kind,'HEAD_AWAY_END')
        self.assertEqual(events[0].duration,3)
        self.assertIsNone(tr.states['HEAD_AWAY']['started'])
    def test_short_glance_does_not_emit(self):
        tr=SustainedEventTracker();self.tick(tr,0,True);self.tick(tr,1,False)
        self.assertFalse(self.tick(tr,1.6,False));self.assertFalse(tr.states['HEAD_AWAY']['emitted'])
    def monitor(self):
        m=VisionMonitor.__new__(VisionMonitor);m.config=Config();m.previous_center=None;m.missing_frames=0
        face=[SimpleNamespace(x=.5,y=.5) for _ in range(478)]
        m.landmarker=SimpleNamespace(detect_for_video=lambda *_:SimpleNamespace(face_landmarks=[face]))
        m.smoother=MeasurementSmoother(.2);m.calibrator=Calibrator(m.config);m.calibrator.ready=True
        m.calibrator.baseline={'yaw':0,'pitch':0,'gaze_x':.5,'gaze_y':.5,'ear':.4}
        m.tracker=SustainedEventTracker();return m
    def test_detected_face_with_bad_pose_does_not_score_as_absence(self):
        m=self.monitor();image=np.zeros((10,10,3),dtype=np.uint8)
        with patch('vision_monitor._measure_face',return_value=None):
            m.process(image,10,10,0,100);r=m.process(image,10,10,3000,103)
        self.assertNotIn('FACE_MISSING',r.active)
        self.assertEqual([(e.kind,e.weight) for e in r.events],[('POSE_UNAVAILABLE',0)])
    def test_small_face_observation_has_no_risk_weight(self):
        m=self.monitor();image=np.zeros((10,10,3),dtype=np.uint8)
        measurement=GazeMeasurement(45,0,.8,.5,.4,.1,(0,0,2,2),((0,0),(1,1)))
        with patch('vision_monitor._measure_face',return_value=measurement):
            m.process(image,10,10,0,100);r=m.process(image,10,10,3000,103)
        self.assertNotIn('HEAD_AWAY',r.active)
        self.assertEqual([(e.kind,e.weight) for e in r.events],[('FACE_TOO_SMALL',0)])
    def test_late_event_keeps_all_accumulated_points(self):
        e=RiskEngine(Config());e.add(RiskEvent(100,'vision','HEAD_AWAY',2));e.refresh(115)
        e.add(RiskEvent(105,'vision','GAZE_AWAY',3))
        self.assertEqual(e.total_points,5)
        self.assertLess(e.score,5)
    def test_modified_escape_stays_modified_after_modifier_release(self):
        stop=threading.Event();m=KeyboardMonitor(queue.Queue(),stop,Config())
        for key in [keyboard.Key.ctrl_l,keyboard.Key.shift,keyboard.Key.esc]:m._on_press(key)
        for key in [keyboard.Key.ctrl_l,keyboard.Key.shift,keyboard.Key.esc]:m._on_release(key)
        self.assertFalse(stop.is_set())
    def test_logged_contribution_includes_confidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'events.csv';logger=EventLogger(p,100)
            logger.write(RiskEvent(101,'vision','HEAD_AWAY',7,confidence=.5),3.5,'NORMAL')
            with p.open(encoding='utf-8-sig') as stream:row=next(csv.DictReader(stream))
            self.assertEqual(float(row['ScoreContribution']),3.5)

if __name__=='__main__':unittest.main()
