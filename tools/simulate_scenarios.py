"""Deterministic rule simulations; no camera, global hooks or clipboard reads."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from models import RiskEvent
from risk_engine import RiskEngine
from vision_monitor import SustainedEventTracker


def run():
    cfg = Config(test_mode=False)
    cases = [
        ('normal_short_head', {'HEAD_AWAY': (3, 4)}, [], 0),
        ('brief_head_with_copy', {'HEAD_AWAY': (3, 4)}, [(3.5, 'keyboard', 'CTRL+C', 2)], 0),
        ('typing_anomaly_only', {}, [(5, 'typing', 'TYPING_SPEED_SPIKE', 7),
                                    (21, 'typing', 'TYPING_SPEED_SPIKE', 7),
                                    (37, 'typing', 'TYPING_SPEED_SPIKE', 7)], 0),
        ('face_missing_45s', {'FACE_MISSING': (3, 48)}, [], 40),
        ('head_away_45s', {'HEAD_AWAY': (3, 48)}, [], 42),
        ('gaze_away_45s', {'GAZE_AWAY': (3, 48)}, [], 48),
        ('multiple_faces_30s', {'MULTIPLE_FACES': (3, 33)}, [], 40),
        ('window_away_45s', {'WINDOW_FOCUS_LOST': (3, 48)}, [], 40),
        ('face_missing_reaches_100', {'FACE_MISSING': (3, 150)}, [], 100),
        ('confirmed_head_with_paste', {'HEAD_AWAY': (3, 10)},
         [(5.5, 'keyboard', 'CTRL+V', 0), (5.5, 'clipboard', 'LARGE_PASTE', 18)], None),
        ('confirmed_head_with_typing', {'HEAD_AWAY': (3, 10)},
         [(5.5, 'typing', 'TYPING_SPEED_SPIKE', 7)], None),
    ]
    specs = {
        'HEAD_AWAY': (cfg.head_min_duration, cfg.head_repeat_sec),
        'GAZE_AWAY': (cfg.gaze_min_duration, cfg.gaze_repeat_sec),
        'FACE_MISSING': (cfg.face_missing_min_duration, cfg.face_missing_repeat_sec),
        'MULTIPLE_FACES': (cfg.multiple_faces_min_duration, cfg.multiple_faces_repeat_sec),
        'WINDOW_FOCUS_LOST': (cfg.focus_lost_min_duration, cfg.focus_lost_repeat_sec),
    }
    summary, trace = [], []
    for name, behaviors, inputs, expected in cases:
        engine, tracker = RiskEngine(cfg), SustainedEventTracker()
        peak, previous_score, first_watch, first_suspicious = 0, 0, None, None
        for step in range(1201):
            elapsed, now = step / 10, 100 + step / 10
            raw = {kind for kind, (begin, end) in behaviors.items() if begin <= elapsed < end}
            events = []
            for kind in behaviors:
                minimum, repeat = specs[kind]
                items = tracker.update(kind=kind, active=kind in raw, now=now,
                                       min_duration=minimum, repeat_sec=repeat,
                                       weight=cfg.event_weights[kind], value='SIMULATION',
                                       confidence=1, context='synthetic scenario')
                if kind == 'WINDOW_FOCUS_LOST':
                    for item in items:
                        item.source = 'window'
                events.extend(items)
            confirmed = {kind for kind in raw if tracker.states[kind]['emitted']}
            engine.set_active_visual(confirmed - {'WINDOW_FOCUS_LOST'})
            durations = {kind: now - tracker.states[kind]['started'] for kind in confirmed}
            engine.set_sustained(durations)
            for when, source, kind, weight in inputs:
                if abs(elapsed - when) < .001:
                    events.append(RiskEvent(now, source, kind, weight))
            for event in events:
                engine.add(event)
            engine.refresh(now)
            engine.check_review()
            assert engine.total_points >= previous_score, (name, 'score must not decrease')
            previous_score = engine.total_points
            peak = max(peak, engine.total_points)
            if engine.state in {'WATCH', 'SUSPICIOUS'} and first_watch is None:
                first_watch = elapsed
            if engine.state == 'SUSPICIOUS' and first_suspicious is None:
                first_suspicious = elapsed
            if step % 10 == 0:
                trace.append([name, elapsed, round(engine.score, 2), round(engine.total_points, 2), engine.reviewed_points, engine.review_pending, engine.state])
        if expected is not None:
            assert abs(peak - expected) < .01, (name, peak, expected)
        engine.set_sustained({})
        engine.refresh(999999)
        assert engine.total_points == peak and engine.score == 0, (name, 'points must remain after behavior ends')
        summary.append(dict(scenario=name, final_score=round(peak, 2), first_watch=first_watch,
                            first_suspicious=first_suspicious, R_after_wait=engine.score, C_preserved=True, review_pending=engine.review_pending))
    out = ROOT / 'validation'
    out.mkdir(exist_ok=True)
    with (out/'review_policy_trace.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Scenario', 'Seconds', 'R', 'C', 'B', 'ReviewPending', 'State'])
        writer.writerows(trace)
    (out/'review_policy_scenarios.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    run()
