"""Local, session-scoped commands from a saved human review to the monitor."""
from __future__ import annotations

import json
import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from reviewer_data import read_snapshot


ACTIONS = {'판단 보류': 'defer', '계속 응시': 'continue', '재시험 요청': 'restart', '시험 중단 결정': 'stop', '퇴장 조치 결정': 'stop'}


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def request_control(live_path: Path, decision_path: Path) -> Path | None:
    record = json.loads(Path(decision_path).read_text(encoding='utf-8'))
    action = ACTIONS.get(record['decision'])
    if not action:
        return None
    live = read_snapshot(live_path)
    if live.session_id != record['session_id'] or any(event.kind == 'SESSION_END' for event in live.events):
        raise ValueError('현재 실행 세션이 아니거나 이미 종료됐습니다. 판단은 저장됐으며 제어 명령은 보내지 않았습니다.')
    # Old CSVs can still be reviewed, but only monitors with an explicit session
    # UUID support commands. Never apply an old decision to a new run.
    if len(live.session_id) != 32:
        raise ValueError('이 로그는 제어 기능 추가 전 세션입니다. 판단은 저장됐습니다. main.py를 다시 실행하면 제어할 수 있습니다.')
    expected = Path(live_path).resolve().parent / 'reviews' / live.session_id / record['review_id'] / 'decision.json'
    if expected.resolve() != Path(decision_path).resolve():
        raise ValueError('현재 세션의 판단 저장 경로와 일치하지 않습니다.')
    command_path = Path(live_path).resolve().parent / 'review_commands' / live.session_id / f"{record['review_id']}.json"
    if command_path.exists() or command_path.with_suffix('.result.json').exists():
        raise ValueError('이 판단의 제어 요청은 이미 전달됐습니다.')
    _write_json(command_path, {'version': 1, 'session_id': live.session_id,
                              'review_id': record['review_id'], 'action': action,
                              'requested_at': datetime.now().astimezone().isoformat(timespec='seconds')})
    return command_path


@dataclass(frozen=True)
class ControlCommand:
    path: Path
    action: str
    reviewer: str
    decision: str
    review_id: str
    reviewed_sequence: int = 0
    reviewed_cumulative: float = 0.0
    reason: str = ""


class ReviewControl:
    def __init__(self, log_path: Path, session_id: str):
        self.log_path = Path(log_path)
        self.root = Path(log_path).resolve().parent
        self.session_id = session_id
        self.directory = self.root / 'review_commands' / session_id
        self.last_poll = -1.0

    def poll(self) -> ControlCommand | None:
        now = time.monotonic()
        if now - self.last_poll < .25:
            return None
        self.last_poll = now
        for path in sorted(self.directory.glob('*.json')):
            if path.name.endswith('.result.json') or path.with_suffix('.result.json').exists():
                continue
            try:
                command = json.loads(path.read_text(encoding='utf-8'))
                review_id = command.get('review_id')
                if (command.get('version') != 1 or command.get('session_id') != self.session_id
                        or review_id != path.stem or command.get('action') not in {'stop', 'restart', 'continue', 'defer'}):
                    raise ValueError('세션 또는 명령 형식 불일치')
                record_path = self.root / 'reviews' / self.session_id / review_id / 'decision.json'
                record = json.loads(record_path.read_text(encoding='utf-8'))
                if (record.get('session_id') != self.session_id
                        or record.get('review_id') != review_id
                        or ACTIONS.get(record.get('decision')) != command['action']
                        or not record.get('reviewer') or not record.get('reason')):
                    raise ValueError('감독관 판단 기록과 명령 불일치')
                evidence_path = record_path.with_name('reviewed_events.csv')
                evidence_bytes = evidence_path.read_bytes()
                if hashlib.sha256(evidence_bytes).hexdigest() != record.get('evidence_sha256'):
                    raise ValueError('검토 근거 해시 불일치')
                evidence, live = read_snapshot(evidence_path), read_snapshot(self.log_path)
                if (live.session_id != self.session_id or evidence.session_id != self.session_id
                        or any(e.kind == 'SESSION_END' for e in live.events)
                        or evidence.warning or evidence.pending
                        or tuple(e.row for e in live.events[:len(evidence.events)]) != tuple(e.row for e in evidence.events)):
                    raise ValueError('현재 세션의 검토 근거 범위 불일치')
                if (record.get('reviewed_sequence', 0) != evidence.through_sequence
                        or record.get('reviewed_cumulative', 0) != evidence.cumulative):
                    raise ValueError('검토 누적값 위조 또는 불일치')
                result = ControlCommand(path, command['action'], record['reviewer'], record['decision'], review_id,
                                        evidence.through_sequence, evidence.cumulative, record['reason'])
                self.complete(result, 'received', '감독관 판단 적용 중')
                return result
            except (OSError, ValueError, KeyError, TypeError) as error:
                _write_json(path.with_suffix('.result.json'), {'status': 'rejected', 'message': str(error)})
        return None

    def complete(self, command: ControlCommand, status: str = 'completed', message: str = '') -> None:
        record = {
            'session_id': self.session_id, 'review_id': command.review_id,
            'action': command.action, 'status': status, 'message': message,
            'updated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        }
        _write_json(command.path.with_suffix('.result.json'), record)
        _write_json(self.root / 'reviews' / self.session_id / command.review_id / 'control_result.json', record)


def command_status(path: Path) -> tuple[str, str]:
    result = path.with_suffix('.result.json')
    if not result.exists():
        return 'pending', '전달 대기 — 실행 중인 모니터가 응답해야 적용됩니다.'
    try:
        record = json.loads(result.read_text(encoding='utf-8'))
        return record['status'], record.get('message', '')
    except (OSError, ValueError, KeyError):
        return 'pending', '처리 결과 읽기 대기'
