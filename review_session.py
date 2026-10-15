"""Nonblocking proctor notification; closing a window never acknowledges evidence."""
from __future__ import annotations

import json
import time
import subprocess

from models import RiskEvent
from activity_monitor import foreground_details


class ReviewSession:
    def __init__(self, engine, logger, controller, opener):
        self.engine, self.logger, self.controller, self.opener = engine, logger, controller, opener
        self.process = None
        self.opened_generation = 0
        self.last_status = -float('inf')
        self.operator_was_active = False

    def log(self, kind, value='', **context):
        self.logger.write(RiskEvent(time.monotonic(), 'system', kind, 0, value=value,
                                   context=json.dumps(context, ensure_ascii=False)),
                          self.engine.score, self.engine.state)

    def operator_active(self):
        return bool(self.process is not None and self.process.poll() is None
                    and foreground_details()[2] == self.process.pid)

    def tick(self, now):
        engine = self.engine
        operator = self.operator_active()
        if operator != self.operator_was_active:
            self.operator_was_active = operator
            self.log('PROCTOR_FOCUS_ENTER' if operator else 'PROCTOR_FOCUS_LEAVE',
                     context='자동 실행한 감독관 창의 키 입력·창 이탈 시간 제외; 영상 감시는 유지')
        if engine.check_review():
            self.log('REVIEW_REQUESTED', str(engine.review_generation),
                     threshold=engine.config.review_threshold, cumulative=engine.total_points,
                     reviewed=engine.reviewed_points, through=engine.reviewed_sequence)
        if engine.review_pending and self.opened_generation != engine.review_generation:
            self.opened_generation = engine.review_generation
            if self.process is None or self.process.poll() is not None:
                self.process = self.opener(self.logger.session_path, control_log_path=self.logger.path)
                if self.process is None:
                    self.log('REVIEW_WINDOW_FAILED', context='확인 대기 유지; reviewer.cmd로 수동 열기')
        if now - self.last_status >= 1.0:
            self.last_status = now
            self.log('SCORE_STATUS')

    def apply(self, command):
        """Return True only for an explicit terminal human decision."""
        self.log('PROCTOR_DECISION', command.decision, reviewer=command.reviewer,
                 reason=command.reason, review_id=command.review_id,
                 reviewed_sequence=command.reviewed_sequence,
                 reviewed_cumulative=command.reviewed_cumulative, action=command.action)
        if command.action in {'stop', 'restart'}:
            return True
        try:
            if command.action == 'continue':
                self.engine.mark_reviewed(command.reviewed_sequence, command.reviewed_cumulative)
                self.log('REVIEW_COMPLETED', command.review_id, reviewer=command.reviewer,
                         reason=command.reason, through=self.engine.reviewed_sequence,
                         reviewed=self.engine.reviewed_points)
            elif command.action == 'defer':
                self.log('REVIEW_DEFERRED', command.review_id, reason=command.reason)
            else:
                raise ValueError('알 수 없는 감독관 조치')
            self.controller.complete(command, message='판단 적용 완료. 시험과 감시를 계속합니다.')
        except ValueError as error:
            self.controller.complete(command, 'rejected', str(error))
        return False

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass
