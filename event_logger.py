from __future__ import annotations

import csv
import shutil
from uuid import uuid4
import threading
from collections import Counter
from datetime import datetime, timedelta
from reviewer_data import FIELDS
from pathlib import Path

from models import RiskEvent


class EventLogger:
    HEADER = list(FIELDS)

    def __init__(self, path: Path, session_start: float, session_id: str | None = None):
        self.path = path
        self.session_id = session_id or uuid4().hex
        self.engine = None
        self.written = set()
        self.wall_start = datetime.now().astimezone()
        self.fallback_total = 0.0
        self.session_start = session_start
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.archived_path: Path | None = None
        if self.path.exists() and self.path.stat().st_size:
            from reviewer_data import read_snapshot
            try:
                previous = read_snapshot(self.path)
                identity = previous.session_id or "unidentified"
            except (OSError, ValueError):
                identity = "unidentified"
            folder = self.path.parent / "sessions"
            folder.mkdir(parents=True, exist_ok=True)
            suffix = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            matching = [p for p in folder.glob(f'{identity}_*.csv') if p.read_bytes() == self.path.read_bytes()]
            self.archived_path = matching[-1] if matching else folder / f"{identity}_{suffix}_{uuid4().hex[:6]}.csv"
            if not matching: shutil.copy2(self.path, self.archived_path)
        folder = self.path.parent / 'sessions'
        folder.mkdir(exist_ok=True)
        self.session_path = folder / f"{self.session_id}_{datetime.now():%Y%m%d-%H%M%S-%f}.csv"
        for target in (self.path, self.session_path):
            with target.open("w", newline="", encoding="utf-8-sig") as stream:
                csv.writer(stream).writerow(self.HEADER)

    def write(self, event: RiskEvent, score: float, state: str) -> None:
        with self.lock:
            if event.event_id in self.written:
                return
            engine = self.engine
            if engine:
                sequence, score, total, contribution = engine.ledger.get(
                    event.event_id, (0, score, engine.total_points, 0.0))
                reviewed, through = engine.reviewed_points, engine.reviewed_sequence
                pending, threshold = int(engine.review_pending), engine.config.review_threshold
            else:
                contribution = event.weight * event.confidence
                self.fallback_total += contribution
                sequence, total = len(self.written) + 1, self.fallback_total
                reviewed, through, pending, threshold = 0.0, 0, 0, 100.0
            elapsed = max(0.0, event.timestamp - self.session_start)
            row = [
                (self.wall_start + timedelta(seconds=elapsed)).isoformat(timespec='milliseconds'),
                f'{elapsed:.3f}', event.source, event.kind, event.value, f'{event.duration:.3f}',
                format(event.confidence, '.17g'), format(contribution, '.17g'),
                format(score, '.17g'), state, event.context,
                self.session_id, event.event_id, sequence, format(event.weight, '.17g'),
                format(total, '.17g'), format(reviewed, '.17g'), through, pending,
                format(threshold, '.17g'), '|'.join(event.related_ids),
            ]
            for target in (self.path, self.session_path):
                with target.open('a', newline='', encoding='utf-8-sig') as stream:
                    csv.writer(stream).writerow(row)
            self.written.add(event.event_id)


def print_summary(engine, keyboard_counts: Counter[str], log_path: Path) -> None:
    print("\n" + "=" * 64)
    print(f"현재 위험도 R={engine.score:.1f} / 누적 C={engine.total_points:.1f} / 검토 완료 B={engine.reviewed_points:.1f}")
    print("=" * 64)
    for name, count in sorted(keyboard_counts.items()):
        print(f"{name:<24}: {count}회")
    print("-" * 64)
    for name, count in sorted(engine.event_counts.items()):
        print(f"{name:<24}: {count}회")
    print("-" * 64)
    print(f"종료 시 위험도 : {engine.score:.1f}")
    print(f"최대 위험도    : {engine.max_score:.1f}")
    print(f"최종 상태      : {engine.state}")
    print(f"로그 파일      : {log_path}")
    print("※ 자동 확정 판정이 아니라 감독관 검토용 보조 지표입니다.")
    print("=" * 64)
