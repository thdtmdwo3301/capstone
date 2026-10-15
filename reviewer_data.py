"""Read monitoring CSVs and persist human review decisions without altering evidence."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4


LEGACY_FIELDS = ('WallTime', 'ElapsedSec', 'Source', 'Event', 'Value', 'DurationSec',
          'Confidence', 'ScoreContribution', 'RiskScore', 'RiskState', 'Context')
FIELDS = LEGACY_FIELDS + ('SessionId', 'EventId', 'EventSequence', 'Weight', 'CumulativeScore', 'ReviewedScore', 'ReviewedThrough', 'ReviewPending', 'ReviewThreshold', 'RelatedEventIds')
DECISIONS = ('판단 보류', '계속 응시', '재시험 요청', '시험 중단 결정', '퇴장 조치 결정')
EVENT_NAMES = {
    'HEAD_AWAY': '고개 이탈', 'GAZE_AWAY': '시선 이탈',
    'FACE_MISSING': '얼굴 미검출', 'MULTIPLE_FACES': '복수 얼굴',
    'WINDOW_FOCUS_LOST': '시험 창 이탈', 'WINDOW_FOCUS_RESTORED': '시험 창 복귀',
    'LARGE_PASTE': '대량 붙여넣기', 'COMBINED_BEHAVIOR': '복합 행동',
    'TYPING_SPEED_SPIKE': '입력 속도 이상', 'TYPING_RHYTHM_ANOMALY': '입력 리듬 이상',
    'LONG_IDLE_THEN_BURST': '정지 후 급입력', 'TYPING_METRICS': '입력 통계',
    'TYPING_REFERENCE_READY': '타이핑 기준 준비', 'CLIPBOARD_CHANGED': '클립보드 변경',
    'CALIBRATION_COMPLETE': '정면 보정 완료', 'CALIBRATION_RETRY': '정면 보정 재시도',
    'CALIBRATION_VERIFYING': '정면 기준 검증', 'CALIBRATION_MANUAL_RESET': '정면 기준 초기화',
    'SESSION_START': '모니터링 시작', 'SESSION_END': '모니터링 종료',
    'REVIEW_REQUESTED': '감독관 호출', 'REVIEW_COMPLETED': '검토 완료',
    'REVIEW_DEFERRED': '판단 보류', 'SCORE_STATUS': '점수 현황',
    'PROCTOR_DECISION': '감독관 조치',
    'RISK_LIMIT_REACHED': '이전 정책의 100점 알림', 'MONITORING_PAUSED': '감독관 확인 대기',
    'EXAM_APP_STARTED': '시험 프로그램 실행', 'EXAM_APP_EXITED': '시험 프로그램 종료 감지',
    'EXAM_APP_STOPPED': '시험 프로그램 종료 완료', 'EXAM_APP_STOP_FAILED': '시험 프로그램 종료 확인 실패',
    'STATE_CHANGE': '경고 단계 변경', 'SUSTAINED_LEVEL': '지속 행동 최소 점수 변경',
    'KEYBOARD_HOOK_FAILED': '키보드 감시 실패', 'KEYBOARD_HOOK_STOPPED': '키보드 감시 중단',
}
STATE_NAMES = {'NORMAL': '정상 범위', 'WATCH': '주의', 'SUSPICIOUS': '추가 검토', 'REVIEW_REQUIRED': '감독관 확인 필요'}


def event_name(kind: str) -> str:
    if kind.endswith('_END') and kind[:-4] in EVENT_NAMES:
        return EVENT_NAMES[kind[:-4]] + ' 종료'
    return EVENT_NAMES.get(kind, kind)


def elapsed_text(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, seconds), 60)
    return f'{int(minutes):02d}:{remainder:04.1f}'


@dataclass(frozen=True)
class LogEvent:
    index: int
    row: dict[str, str]

    @property
    def elapsed(self) -> float:
        return float(self.row['ElapsedSec'])

    @property
    def score(self) -> float:
        return float(self.row['RiskScore'])

    @property
    def cumulative(self) -> float:
        return float(self.row.get('CumulativeScore', '0'))

    @property
    def reviewed(self) -> float:
        return float(self.row.get('ReviewedScore', '0'))

    @property
    def contribution(self) -> float:
        return float(self.row['ScoreContribution'])

    @property
    def kind(self) -> str:
        return self.row['Event']

    @property
    def needs_review(self) -> bool:
        return self.contribution > 0 or self.kind in {
            'STATE_CHANGE', 'SUSTAINED_LEVEL', 'KEYBOARD_HOOK_FAILED', 'KEYBOARD_HOOK_STOPPED',
            'REVIEW_REQUESTED', 'REVIEW_COMPLETED', 'REVIEW_DEFERRED', 'RISK_LIMIT_REACHED', 'EXAM_APP_STOPPED', 'EXAM_APP_STOP_FAILED', 'PROCTOR_DECISION'
        }


@dataclass(frozen=True)
class LogSnapshot:
    path: Path
    session_id: str
    started_at: str
    events: tuple[LogEvent, ...]
    warning: str = ''
    pending: bool = False

    @property
    def peak(self) -> float:
        return max((event.score for event in self.events), default=0.0)

    @property
    def latest(self) -> LogEvent | None:
        return self.events[-1] if self.events else None

    @property
    def counts(self) -> Counter:
        return Counter(event.kind for event in self.events if event.contribution > 0)

    @property
    def cumulative(self):
        return self.latest.cumulative if self.latest else 0.0

    @property
    def through_sequence(self):
        return max((int(e.row.get('EventSequence', '0')) for e in self.events), default=0)

    def csv_bytes(self) -> bytes:
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=tuple(self.events[0].row) if self.events else FIELDS)
        writer.writeheader()
        writer.writerows(event.row for event in self.events)
        return stream.getvalue().encode('utf-8-sig')


def read_snapshot(path: Path) -> LogSnapshot:
    path = Path(path).resolve()
    if not path.exists():
        return LogSnapshot(path, '', '', (), '로그 파일 생성 대기 중')
    raw = path.read_bytes()
    # Writers append complete CSV records ending in a newline. Do not display a
    # partial final record, including incomplete UTF-8 or a quoted context field.
    end = raw.rfind(b'\n') + 1
    pending = end < len(raw)
    if not end:
        return LogSnapshot(path, '', '', (), '로그 기록 대기 중', bool(raw))
    text = raw[:end].decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
    if reader.fieldnames not in (list(FIELDS), list(LEGACY_FIELDS)):
        raise ValueError('모니터링 CSV 형식이 아닙니다. anti_cheating_events.csv를 선택하세요.')
    events: list[LogEvent] = []
    invalid = 0
    try:
        for row in reader:
            if None in row or any(row.get(field) is None for field in reader.fieldnames):
                invalid += 1
                continue
            try:
                numbers = [float(row[key]) for key in ('ElapsedSec', 'DurationSec', 'Confidence',
                                                      'ScoreContribution', 'RiskScore')]
                if not all(math.isfinite(value) and value >= 0 for value in numbers):
                    raise ValueError('non-finite or negative number')
                if numbers[2] > 1 or not row['Event'] or not row['WallTime']:
                    raise ValueError('invalid field')
                if 'CumulativeScore' in row:
                    extra = [float(row[k]) for k in ('Weight', 'CumulativeScore', 'ReviewedScore', 'ReviewThreshold')]
                    if not all(math.isfinite(v) and v >= 0 for v in extra):
                        raise ValueError('invalid ledger number')
                    if (extra[2] > extra[1] + 1e-6 or extra[3] <= 0
                            or int(row['EventSequence']) < 0 or int(row['ReviewedThrough']) < 0
                            or row['ReviewPending'] not in {'0','1'} or not row['EventId']):
                        raise ValueError('invalid review watermark')
                    if events and extra[1] + 1e-6 < events[-1].cumulative:
                        raise ValueError('cumulative score decreased')
            except ValueError:
                invalid += 1
                continue
            events.append(LogEvent(len(events) + 1, row))
    except csv.Error:
        if pending:
            pass  # A multiline final record can still be in flight.
        else:
            raise ValueError('CSV 인용부호 또는 행 형식이 손상됐습니다.')
    # A valid monitoring file contains one session. Reject concatenated sessions
    # instead of combining evidence belonging to different examinees.
    starts = [event for event in events if event.kind == 'SESSION_START']
    if len(starts) > 1:
        raise ValueError('여러 세션이 합쳐진 CSV입니다. 세션별 로그를 선택하세요.')
    anchor = starts[0] if starts else (events[0] if events else None)
    identity = json.dumps(anchor.row, ensure_ascii=False, sort_keys=True) if anchor else ''
    session_id = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20] if anchor else ''
    if anchor and anchor.kind == 'SESSION_START' and re.fullmatch(r'[0-9a-f]{32}', anchor.row['Value']):
        session_id = anchor.row['Value']
    if anchor and 'SessionId' in anchor.row:
        explicit = anchor.row['SessionId']
        if not re.fullmatch(r'[0-9a-f]{32}', explicit):
            raise ValueError('잘못된 세션 ID입니다.')
        if (anchor.kind == 'SESSION_START' and re.fullmatch(r'[0-9a-f]{32}', anchor.row['Value'])
                and anchor.row['Value'] != explicit):
            raise ValueError('세션 시작 기록과 세션 ID가 다릅니다.')
        session_id = explicit
    if any(e.row.get('SessionId', session_id) != session_id for e in events):
        raise ValueError('사건의 세션 ID가 일치하지 않습니다.')
    warning = f'형식이 잘못된 {invalid}개 행을 제외했습니다.' if invalid else ''
    return LogSnapshot(path, session_id, anchor.row['WallTime'] if anchor else '',
                       tuple(events), warning, pending)


def save_decision(snapshot: LogSnapshot, directory: Path, *, reviewer: str,
                  decision: str, reason: str, examinee: str = '',
                  selected_indices: tuple[int, ...] = ()) -> Path:
    reviewer, reason = reviewer.strip(), reason.strip()
    if not snapshot.session_id or not snapshot.events:
        raise ValueError('검토할 로그가 아직 없습니다.')
    if not reviewer or not reason:
        raise ValueError('감독관 이름과 판단 사유를 입력하세요.')
    if decision not in DECISIONS:
        raise ValueError('지원하지 않는 판단입니다.')
    if snapshot.warning or snapshot.pending:
        raise ValueError('손상되거나 기록 중인 근거로 검토를 완료할 수 없습니다. 다시 읽으세요.')
    available = {event.index for event in snapshot.events}
    if not set(selected_indices).issubset(available):
        raise ValueError('선택한 사건이 현재 검토 세션에 없습니다.')
    latest = read_snapshot(snapshot.path)
    # A live append is fine. An overwritten log or edited evidence is not.
    if (latest.session_id != snapshot.session_id
            or tuple(event.row for event in latest.events[:len(snapshot.events)])
            != tuple(event.row for event in snapshot.events)):
        raise ValueError('검토 중 원본 세션이 변경됐습니다. 새로고침 후 다시 확인하세요.')
    evidence = snapshot.csv_bytes()
    record = {
        'schema_version': 2,
        'review_id': str(uuid4()),
        'reviewed_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'session_id': snapshot.session_id,
        'session_started_at': snapshot.started_at,
        'source_path': str(snapshot.path),
        'reviewer': reviewer,
        'examinee': examinee.strip(),
        'decision': decision,
        'reason': reason,
        'selected_event_indices': sorted(set(selected_indices)),
        'reviewed_event_count': len(snapshot.events),
        'reviewed_through': snapshot.events[-1].row['WallTime'],
        'peak_recorded_risk': snapshot.peak,
        'reviewed_sequence': snapshot.through_sequence,
        'reviewed_cumulative': snapshot.cumulative,
        'previous_reviewed_score': snapshot.latest.reviewed,
        'reviewed_event_ids': [e.row.get('EventId', '') for e in snapshot.events],
        'review_call_times': [e.row['WallTime'] for e in snapshot.events if e.kind == 'REVIEW_REQUESTED'],
        'evidence_sha256': hashlib.sha256(evidence).hexdigest(),
        'evidence_file': 'reviewed_events.csv',
        'parse_warning': snapshot.warning,
        'exam_control_executed': False,
        'managed_exam_app_stopped': any(e.kind == 'EXAM_APP_STOPPED' for e in snapshot.events),
    }
    folder = Path(directory) / snapshot.session_id / record['review_id']
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'reviewed_events.csv').write_bytes(evidence)
    # Publishing decision.json last makes incomplete saves invisible to history.
    target = folder / 'decision.json'
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return target


def decision_history(directory: Path, session_id: str) -> list[dict]:
    if not session_id:
        return []
    records = []
    for path in (Path(directory) / session_id).glob('*/decision.json'):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('session_id') == session_id:
                records.append(record)
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda record: record.get('reviewed_at', ''), reverse=True)
