from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import replace
from config import Config
from models import RiskEvent

OBSERVATIONS = {'CTRL+C', 'CTRL+X', 'CTRL+V', 'SHIFT+INSERT', 'ALT+TAB',
                'WIN+TAB', 'PRINT_SCREEN', 'WIN+SHIFT+S'}

class RiskEngine:
    """R is a decaying display. C and reviewed watermark B form an audit ledger."""
    def __init__(self, config: Config):
        self.config = config
        self.events = deque()
        self.state = 'NORMAL'
        self.score = self.total_points = self.max_score = 0.0
        self.reviewed_points = 0.0
        self.reviewed_sequence = 0
        self.review_pending = False
        self.review_generation = 0
        self.event_counts = Counter()
        self.active_visual = set()
        self.last_combo_at = -math.inf
        self.sustained_durations = {}
        self.sustained_floor = 0.0
        self.sustained_reasons = ()
        self.latest_time = 0.0
        self.seen = set()
        self.combined_pairs = set()
        # id -> (sequence, R, C, contribution), captured at each acceptance.
        self.ledger = {}
        self.sequence_totals = {0: 0.0}

    @property
    def unreviewed_points(self):
        return max(0.0, self.total_points - self.reviewed_points)

    def check_review(self):
        if not self.review_pending and self.unreviewed_points >= self.config.review_threshold:
            self.review_pending = True
            self.review_generation += 1
            return True
        return False

    def mark_reviewed(self, sequence: int, cumulative: float):
        expected = self.sequence_totals.get(sequence)
        if (sequence <= self.reviewed_sequence or expected is None
                or not math.isclose(expected, cumulative, abs_tol=1e-6)):
            raise ValueError('검토 사건 범위와 누적값이 일치하지 않거나 이미 처리한 범위입니다.')
        self.reviewed_sequence = sequence
        self.reviewed_points = expected
        self.review_pending = False

    def set_active_visual(self, active):
        self.active_visual = set(active)

    def set_sustained(self, durations):
        self.sustained_durations = dict(durations)

    def _typing_support(self, event):
        if event.source != 'typing' or event.weight <= 0:
            return event
        support = [e.event_id for e in self.events if e.kind == 'WINDOW_FOCUS_LOST'
                   and 0 <= event.timestamp - e.timestamp <= self.config.combo_window_sec]
        confirmed = bool(self.active_visual) or bool(support)
        return replace(event, weight=min(event.weight, 2.0) if confirmed else 0.0,
                       related_ids=tuple(support[-1:]),
                       context=event.context + ('; 보조 최대 2점; confirmed_visual=' + ','.join(sorted(self.active_visual))
                                                if confirmed else '; 단독 관찰 0점'))

    @staticmethod
    def _is_visual(event):
        return event.source == 'vision' and event.weight * event.confidence > 0

    @staticmethod
    def _is_input(event):
        return event.kind in {'LARGE_PASTE', 'WINDOW_FOCUS_LOST'} and event.weight * event.confidence > 0

    def _recent_opposite(self, event):
        predicate = self._is_visual if self._is_input(event) else self._is_input if self._is_visual(event) else None
        if predicate:
            return next((e for e in sorted(self.events, key=lambda x: x.timestamp, reverse=True)
                         if 0 <= event.timestamp-e.timestamp <= self.config.combo_window_sec and predicate(e)), None)
        return None

    def _calculate(self, now):
        self.latest_time = max(now, self.latest_time)
        self.events = deque(e for e in self.events if self.latest_time-e.timestamp <=
                            max(self.config.score_window_sec, self.config.combo_window_sec))
        return sum(e.weight * e.confidence * 2 ** (-(self.latest_time-e.timestamp)/self.config.score_half_life_sec)
                   for e in self.events if 0 <= self.latest_time-e.timestamp <= self.config.score_window_sec)

    def _transition(self):
        previous = self.state
        if self.score >= self.config.suspicious_enter:
            self.state = 'SUSPICIOUS'
        elif self.score < self.config.watch_exit:
            self.state = 'NORMAL'
        elif self.state == 'SUSPICIOUS' and self.score < self.config.suspicious_exit:
            self.state = 'WATCH'
        elif self.state == 'NORMAL' and self.score >= self.config.watch_enter:
            self.state = 'WATCH'
        return (previous, self.state) if previous != self.state else None

    def _accept(self, event):
        self.seen.add(event.event_id)
        contribution = event.weight * event.confidence
        if contribution > 0:
            self.events.append(event)
            self.event_counts[event.kind] += 1
        self.total_points += contribution
        self.score = self._calculate(event.timestamp)
        self.max_score = max(self.max_score, self.score)
        sequence = len(self.ledger) + 1
        self.ledger[event.event_id] = (sequence, self.score, self.total_points, contribution)
        self.sequence_totals[sequence] = self.total_points

    def add(self, event):
        if event.event_id in self.seen:
            return [], None
        if not event.event_id or not all(math.isfinite(v) for v in
                                        (event.timestamp, event.weight, event.confidence, event.duration)):
            raise ValueError('잘못된 사건 ID 또는 숫자')
        if event.weight < 0 or not 0 <= event.confidence <= 1 or event.duration < 0:
            raise ValueError('잘못된 가중치·신뢰계수·지속시간')
        if event.kind in OBSERVATIONS:
            event = replace(event, weight=0.0)
        event = self._typing_support(event)
        opposite = self._recent_opposite(event)
        accepted = [event]
        self._accept(event)
        pair = tuple(sorted((event.event_id, opposite.event_id))) if opposite else ()
        if (opposite is not None and pair not in self.combined_pairs
                and event.timestamp-self.last_combo_at >= self.config.combo_cooldown_sec):
            combo = RiskEvent(event.timestamp, 'fusion', 'COMBINED_BEHAVIOR',
                              self.config.event_weights['COMBINED_BEHAVIOR'],
                              value=f'{event.kind}+{opposite.kind}',
                              confidence=min(event.confidence, opposite.confidence),
                              context='5초 이내 영상·입력 사건 결합; 같은 사건 쌍 재처리 금지',
                              event_id='combo:' + ':'.join(pair), related_ids=pair)
            self.combined_pairs.add(pair)
            self.last_combo_at = event.timestamp
            self._accept(combo)
            accepted.append(combo)
        return accepted, self._transition()

    def refresh(self, now):
        self.score = self._calculate(now)
        self.max_score = max(self.max_score, self.score)
        return self._transition()

