"""Local proctor review window: live CSV evidence and explicit human decisions."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import Config
from reviewer_data import (DECISIONS, STATE_NAMES, LogSnapshot, decision_history,
                           elapsed_text, event_name, read_snapshot, save_decision)
from review_control import command_status, request_control


def control_log_path_for(log_path: Path, explicit: Path | None = None) -> Path:
    """Evidence can live in sessions/; controls belong to the live monitor root."""
    source = Path(log_path).resolve()
    if explicit is not None:
        return Path(explicit).resolve()
    if source.parent.name == 'sessions':
        # Compatibility with already-running monitors from before --control-log.
        configured = Config().log_path.resolve()
        if configured.parent == source.parent.parent:
            return configured
        # A standalone archive is readable, but session checks still prevent
        # sending its decision to a different or nonexistent current session.
        return source.parent.parent / 'anti_cheating_events.csv'
    return source


class ReviewerApp:
    def __init__(self, root: tk.Tk, log_path: Path, auto_review=False,
                 control_log_path: Path | None = None):
        self.root = root
        self.live_path = control_log_path_for(log_path, control_log_path)
        self.path = Path(log_path).resolve()
        self.review_dir = self.live_path.parent / 'reviews'
        self.snapshot: LogSnapshot | None = None
        self.review_snapshot: LogSnapshot | None = None
        self.signature = None
        self.pending_command: Path | None = None
        self.last_alert_id = None
        self.closed = False
        self.auto_review = auto_review
        self.follow = tk.BooleanVar(value=True)
        self.filter = tk.StringVar(value='검토 이벤트')
        self.notice = tk.StringVar(value='로그를 읽는 중입니다.')
        self.command_notice = tk.StringVar(value='감독관의 판단을 저장하면 선택한 조치를 적용합니다.')
        self.reviewer = tk.StringVar()
        self.examinee = tk.StringVar()
        self.decision = tk.StringVar(value='판단 보류')
        self.root.title('캡스톤 · 감독관 확인창')
        width = min(1280, self.root.winfo_screenwidth() - 70)
        height = min(850, self.root.winfo_screenheight() - 110)
        self.root.geometry(f'{width}x{height}+25+25')
        self.root.minsize(1000, 680)
        self.root.configure(background='#eef2f7')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('.', font=('Malgun Gothic', 10))
        style.configure('TFrame', background='#eef2f7')
        style.configure('TLabel', background='#eef2f7', foreground='#18263d')
        style.configure('Title.TLabel', font=('Malgun Gothic', 19, 'bold'))
        style.configure('Card.TLabel', font=('Malgun Gothic', 17, 'bold'))
        style.configure('Treeview', rowheight=29, font=('Malgun Gothic', 10))
        style.configure('Treeview.Heading', font=('Malgun Gothic', 10, 'bold'))
        style.configure('Action.TButton', padding=(12, 10), font=('Malgun Gothic', 11, 'bold'))
        self._build()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self._poll()

    def _build(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='감독관 확인창', style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer, text='기준 도달 시 확인 대기 · 시험과 감시는 계속됩니다. 점수는 부정행위 확정이나 확률이 아닙니다.').pack(anchor='w', pady=(3, 10))
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='현재 세션 보기', command=self._open_live).pack(side='left')
        ttk.Button(toolbar, text='이전 CSV 열기', command=self._choose_file).pack(side='left', padx=6)
        ttk.Checkbutton(toolbar, text='1초마다 갱신', variable=self.follow).pack(side='left', padx=8)
        ttk.Button(toolbar, text='새로고침', command=lambda: self._refresh(True)).pack(side='left')
        ttk.Button(toolbar, text='새 근거까지 검토 범위에 포함', command=self._include_latest).pack(side='left', padx=8)
        self.scope_notice = tk.StringVar(value='검토 범위 준비 중')
        ttk.Label(outer, textvariable=self.scope_notice, foreground='#175a86').pack(anchor='w', pady=4)
        self.path_label = ttk.Label(outer, text='', wraplength=1180)
        self.path_label.pack(anchor='w', pady=(7, 8))
        cards = ttk.Frame(outer)
        cards.pack(fill='x', pady=(0, 8))
        self.cards = []
        for title in ('현재 위험도 R', '세션 누적 C', '미검토 C − B / 기준 T', '검토 상태 · 완료 B'):
            card = ttk.LabelFrame(cards, text=title, padding=9)
            card.pack(side='left', fill='x', expand=True, padx=(0, 8))
            label = ttk.Label(card, text='—', style='Card.TLabel')
            label.pack(anchor='w')
            self.cards.append(label)
        ttk.Label(outer, textvariable=self.notice, wraplength=1200).pack(side='bottom', anchor='w', pady=(10, 0))
        panes = ttk.Panedwindow(outer, orient='horizontal')
        panes.pack(fill='both', expand=True)
        left, right = ttk.Frame(panes), ttk.Frame(panes, padding=(16, 0, 0, 0))
        panes.add(left, weight=3)
        panes.add(right, weight=1)
        review_tabs = ttk.Notebook(right)
        review_tabs.pack(fill='both', expand=True)
        decision_panel = ttk.Frame(review_tabs, padding=6)
        history_panel = ttk.Frame(review_tabs, padding=6)
        review_tabs.add(decision_panel, text='감독관 판단')
        review_tabs.add(history_panel, text='판단 이력')
        right = decision_panel
        self.chart = tk.Canvas(left, height=110, background='white', highlightthickness=0)
        self.chart.pack(fill='x')
        self.chart.bind('<Configure>', lambda _: self._draw_chart())
        ttk.Label(left, text='파랑 R · 주황 C · 세로축 자동 조정. 기록 시점 값이며 정확도·확률이 아닙니다.', font=('Malgun Gothic', 9)).pack(anchor='w', pady=(3, 6))
        filters = ttk.Frame(left)
        filters.pack(fill='x', pady=(0, 6))
        ttk.Label(filters, text='사건 목록').pack(side='left')
        selector = ttk.Combobox(filters, state='readonly', width=16, textvariable=self.filter,
                                values=('검토 이벤트', '점수 가산 이벤트', '전체 기록'))
        selector.pack(side='right')
        selector.bind('<<ComboboxSelected>>', lambda _: self._fill_events())
        table_frame = ttk.Frame(left)
        table_frame.pack(fill='both', expand=True)
        columns = ('time', 'event', 'value', 'duration', 'contribution', 'risk', 'cumulative')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', selectmode='extended', height=4)
        for key, label, width in zip(columns, ('경과', '감지 행동', '방향·값', '지속(초)', '기여점수', '위험도 R', '누적 C'), (60, 150, 65, 55, 65, 65, 70)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, minwidth=50, stretch=key == 'event')
        scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.tree.tag_configure('SUSPICIOUS', background='#ffe4e6')
        self.tree.tag_configure('REVIEW_REQUIRED', background='#ffd5d9')
        self.tree.tag_configure('WATCH', background='#fff2d6')
        self.tree.bind('<<TreeviewSelect>>', lambda _: self._show_event())
        ttk.Label(left, text='선택 사건의 근거 · Ctrl/Shift로 여러 사건을 선택할 수 있습니다.').pack(anchor='w', pady=(9, 4))
        self.details = tk.Text(left, width=60, height=6, wrap='word', font=('Malgun Gothic', 10), relief='flat', padx=9, pady=7)
        self.details.pack(fill='x')
        self.details.configure(state='disabled')
        ttk.Label(right, text='감독관 판단', font=('Malgun Gothic', 13, 'bold')).pack(anchor='w')
        for label, variable in (('감독관 이름', self.reviewer), ('응시자·시험 식별 (선택)', self.examinee)):
            ttk.Label(right, text=label).pack(anchor='w', pady=(6, 2))
            ttk.Entry(right, textvariable=variable).pack(fill='x')
        ttk.Label(right, text='판단').pack(anchor='w', pady=(6, 2))
        ttk.Combobox(right, values=DECISIONS, state='readonly', textvariable=self.decision).pack(fill='x')
        ttk.Label(right, text='근거와 사유').pack(anchor='w', pady=(6, 2))
        self.reason = tk.Text(right, width=32, height=4, wrap='word', font=('Malgun Gothic', 10), relief='flat', padx=7, pady=6)
        self.reason.pack(fill='x')
        ttk.Label(right, text='재시험: 같은 환경에서 새 세션을 0점으로 시작\n중단·퇴장: 판단을 기록하고 종료\n계속 응시: 고정된 검토 범위만 완료\n창 닫기·새로고침: 검토 완료 아님', wraplength=310, font=('Malgun Gothic', 9)).pack(anchor='w', pady=8)
        self.save_button = ttk.Button(right, text='판단 저장 · 조치 적용', style='Action.TButton', command=self._save)
        self.save_button.pack(fill='x')
        ttk.Label(right, textvariable=self.command_notice, wraplength=285, foreground='#175a86').pack(anchor='w', pady=8)
        ttk.Label(history_panel, text='이 세션의 판단 이력', font=('Malgun Gothic', 11, 'bold')).pack(anchor='w', pady=(9, 4))
        self.history = tk.Text(history_panel, width=32, height=4, wrap='word', font=('Malgun Gothic', 9), relief='flat', padx=7, pady=6)
        self.history.pack(fill='both', expand=True)
        self.history.configure(state='disabled')

    @staticmethod
    def _set_text(widget, text):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def _allow_switch(self):
        if self.reason.get('1.0', 'end-1c').strip():
            return messagebox.askyesno('검토 세션 변경', '작성 중인 사유를 지우고 다른 로그를 여시겠습니까?', parent=self.root)
        return True

    def _open_live(self):
        if self._allow_switch():
            self._switch(self.live_path)

    def _choose_file(self):
        if not self._allow_switch():
            return
        selected = filedialog.askopenfilename(parent=self.root, title='검토할 모니터링 CSV',
                    initialdir=str(self.live_path.parent), filetypes=[('CSV 기록', '*.csv')])
        if selected:
            self._switch(Path(selected))

    def _switch(self, path):
        self.path = Path(path).resolve()
        self.snapshot = None
        self.review_snapshot = None
        self.signature = None
        self.reason.delete('1.0', 'end')
        self.examinee.set('')
        self.decision.set('판단 보류')
        self.follow.set(self.path == self.live_path)
        self._refresh(True)

    def _include_latest(self):
        if self.pending_command or self.snapshot is None:
            return
        self.review_snapshot = self.snapshot
        self._scope_label()
        self._fill_events()

    def _scope_label(self):
        cut, live = self.review_snapshot, self.snapshot
        if not cut or not live:
            return
        self.scope_notice.set(f'이번 검토: 사건 순번 1–{cut.through_sequence}, C={cut.cumulative:.2f}까지 고정 | '
                              f'검토 중 추가된 누적분 {max(0, live.cumulative-cut.cumulative):.2f}점은 미포함')

    def _refresh(self, force=False):
        try:
            stat = self.path.stat() if self.path.exists() else None
            signature = (str(self.path), stat.st_size, stat.st_mtime_ns) if stat else (str(self.path), None)
            if not force and signature == self.signature:
                return
            incoming = read_snapshot(self.path)
            if self.snapshot and incoming.session_id != self.snapshot.session_id:
                if self.reason.get('1.0', 'end-1c').strip():
                    archives = sorted((self.live_path.parent / 'sessions').glob(f'{self.snapshot.session_id}_*.csv'))
                    if archives:
                        self.path = archives[-1]
                        self.snapshot = replace(self.snapshot, path=self.path)
                    self.follow.set(False)
                    self.signature = None
                    self.notice.set('새 세션이 시작됐습니다. 작성 중인 검토는 유지했습니다. 완료 후 현재 세션 보기를 누르세요.')
                    self.path_label.configure(text=f'검토 보류 중: {self.path}')
                    return
                self.examinee.set('')
                self.decision.set('판단 보류')
                self.tree.selection_remove(self.tree.selection())
            self.signature = signature
            self.snapshot = incoming
            alerts = [e for e in incoming.events if e.kind == 'REVIEW_REQUESTED']
            alert_id = alerts[-1].row.get('EventId') if alerts else None
            if self.last_alert_id is not None and alert_id != self.last_alert_id:
                self.root.deiconify()
                self.root.lift()
                self.root.bell()
            self.last_alert_id = alert_id
            if self.review_snapshot is None or self.review_snapshot.session_id != incoming.session_id:
                self.review_snapshot = incoming
            self._scope_label()
            self.path_label.configure(text=f'세션 시작: {incoming.started_at or "대기"}  |  {self.path}')
            latest = incoming.latest
            self.cards[0].configure(text=f'{latest.score:.1f}' if latest else '—')
            modern = latest and 'CumulativeScore' in latest.row
            self.cards[1].configure(text=f'{incoming.cumulative:.1f}' if modern else '이전 형식')
            unreviewed = max(0.0, incoming.cumulative-latest.reviewed) if modern else 0
            threshold = float(latest.row['ReviewThreshold']) if modern else 100
            self.cards[2].configure(text=f'{unreviewed:.1f} / {threshold:g}' if modern else '—')
            state = '확인 대기' if modern and latest.row['ReviewPending'] == '1' else '감시 중'
            self.cards[3].configure(text=f'{state} · B={latest.reviewed:.1f}' if modern else '이전 기록')
            ended = any(e.kind == 'SESSION_END' for e in incoming.events)
            label = '종료 기록 있음' if ended else '마지막 기록: ' + (latest.row['WallTime'] if latest else '없음')
            self.notice.set(f'{label}  ·  {len(incoming.events)}개 기록  {incoming.warning}' + (' · 마지막 행 기록 중' if incoming.pending else ''))
            self._fill_events()
            self._draw_chart()
            self._history()
        except (OSError, ValueError, UnicodeError) as error:
            self.notice.set(f'로그 읽기 대기: {error}')

    def _fill_events(self):
        selected = self.tree.selection()
        view = self.tree.yview()
        self.tree.delete(*self.tree.get_children())
        if not self.snapshot:
            return
        events = self.snapshot.events
        if self.filter.get() == '검토 이벤트':
            events = tuple(e for e in events if e.needs_review)
        elif self.filter.get() == '점수 가산 이벤트':
            events = tuple(e for e in events if e.contribution > 0)
        for event in reversed(events):
            row = event.row
            self.tree.insert('', 'end', iid=str(event.index), values=(elapsed_text(event.elapsed), ('[미포함] ' if self.review_snapshot and event.index > len(self.review_snapshot.events) else '') + event_name(event.kind),
                             row['Value'][:32], f"{float(row['DurationSec']):.1f}", f'{event.contribution:.2f}',
                             f'{event.score:.1f}', f'{event.cumulative:.1f}'), tags=(row['RiskState'],))
        keep = [item for item in selected if self.tree.exists(item)]
        if keep:
            self.tree.selection_set(keep)
        self.tree.yview_moveto(view[0] if keep and view else 0)
        self._show_event()

    def _show_event(self):
        selection = self.tree.selection()
        if not selection or not self.snapshot:
            self._set_text(self.details, '사건을 선택하면 발생 시각, 지속시간, 방향·값과 원본 근거를 표시합니다.\n기여점수는 해당 이벤트 발생 시점의 값이며 현재 총점이나 부정행위 확률이 아닙니다.')
            return
        event = self.snapshot.events[int(selection[-1]) - 1]
        row = event.row
        text = (f"선택 {len(selection)}건 · {row['WallTime']} · 경과 {elapsed_text(event.elapsed)}\n"
                f"{event_name(event.kind)} ({event.kind}) | 값: {row['Value']} | 지속 {row['DurationSec']}초\n"
                f"이벤트 기여 {event.contribution:.2f}점 | 당시 위험도 {event.score:.1f} | {STATE_NAMES.get(row['RiskState'], row['RiskState'])}\n"
                f"사건 ID: {row.get('EventId', '이전 형식')} | 결합 ID: {row.get('RelatedEventIds', '')}\n"
                f"C={event.cumulative:.2f} / B={event.reviewed:.2f} / 가중치={row.get('Weight', '이전 형식')} | 보정 계수: {row['Confidence']} (부정행위 확률 아님)\n근거: {row['Context']}")
        self._set_text(self.details, text)

    def _draw_chart(self):
        self.chart.delete('all')
        width = max(self.chart.winfo_width(), 300)
        height = 110
        left, right, top, bottom = 34, width - 12, 8, height - 20
        maximum = max(100.0, max((max(e.score, e.cumulative) for e in self.snapshot.events), default=0.0)) if self.snapshot else 100.0
        for value in (0, maximum/2, maximum):
            y = bottom - (bottom - top) * value / maximum
            self.chart.create_line(left, y, right, y, fill='#e5eaf1')
            self.chart.create_text(left - 7, y, text=f'{value:g}', anchor='e', font=('Malgun Gothic', 8), fill='#68758a')
        if not self.snapshot or not self.snapshot.events:
            return
        events = sorted(self.snapshot.events, key=lambda e: e.elapsed)
        duration = max(1.0, events[-1].elapsed)
        points = []
        for event in events:
            points.extend((left + (right - left) * event.elapsed / duration, bottom - (bottom - top) * event.score / maximum))
        if len(points) >= 4:
            self.chart.create_line(*points, fill='#2668b0', width=2)
        else:
            x, y = points
            self.chart.create_oval(x - 2, y - 2, x + 2, y + 2, fill='#2668b0', outline='')
        cumulative = []
        for event in events:
            cumulative.extend((left+(right-left)*event.elapsed/duration, bottom-(bottom-top)*event.cumulative/maximum))
        if len(cumulative) >= 4:
            self.chart.create_line(*cumulative, fill='#c76b16', width=2)
        self.chart.create_text(left, height - 7, text='00:00', anchor='w', font=('Malgun Gothic', 8))
        self.chart.create_text(right, height - 7, text=elapsed_text(duration), anchor='e', font=('Malgun Gothic', 8))

    def _history(self):
        records = decision_history(self.review_dir, self.snapshot.session_id if self.snapshot else '')
        lines = [f"{r['reviewed_at']}\n{r['reviewer']} · {r['decision']}\n{r['reason']}" for r in records]
        self._set_text(self.history, '\n\n'.join(lines) or '저장된 판단이 없습니다.')

    def _save(self):
        if self.snapshot is None:
            return
        if self.pending_command:
            messagebox.showinfo('조치 처리 중', '이전 제어 요청의 처리 결과를 기다리고 있습니다.', parent=self.root)
            return
        try:
            saved = save_decision(self.review_snapshot, self.review_dir, reviewer=self.reviewer.get(),
                                  examinee=self.examinee.get(), decision=self.decision.get(),
                                  reason=self.reason.get('1.0', 'end-1c'),
                                  selected_indices=tuple(int(item) for item in self.tree.selection()))
        except (OSError, ValueError) as error:
            messagebox.showerror('판단 저장 실패', str(error), parent=self.root)
            return
        self.reason.delete('1.0', 'end')
        self._history()
        try:
            self.pending_command = request_control(self.live_path, saved)
        except (OSError, ValueError) as error:
            self.command_notice.set(str(error))
            return
        if self.pending_command:
            self.command_notice.set('판단 저장 완료 · 모니터링 제어 요청 전달 대기')
        else:
            self.command_notice.set('판단과 검토한 CSV 근거를 저장했습니다.')

    def _poll(self):
        if self.closed:
            return
        if self.follow.get() or self.snapshot is None:
            self._refresh()
        if self.pending_command:
            status, message = command_status(self.pending_command)
            self.command_notice.set(message)
            if status in {'completed', 'rejected', 'error'}:
                self.pending_command = None
                if status == 'completed':
                    self.command_notice.set(message + ' 다음 사건은 검토 범위 포함 버튼으로 확인하세요.')
        self.root.after(1000, self._poll)

    def close(self):
        self.closed = True
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description='감독관 CSV 검토창')
    parser.add_argument('--log', type=Path, default=Config().log_path)
    parser.add_argument('--control-log', type=Path,
                        help='실행 중인 감시기의 현재 CSV 경로; 검토 근거 경로와 별도로 지정')
    parser.add_argument('--auto-review', action='store_true')
    arguments = parser.parse_args()
    root = tk.Tk()
    ReviewerApp(root, arguments.log, arguments.auto_review, arguments.control_log)
    root.mainloop()


if __name__ == '__main__':
    main()
