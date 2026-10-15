from __future__ import annotations

import queue
import statistics
import threading
import time
from collections import Counter, deque

from pynput import keyboard

from activity_monitor import clipboard_metadata
from config import Config
from models import RiskEvent, TypingSnapshot
from typing_setup import TypingReference


CONTROL_CHARS = {
    "\x03": "c", "\x16": "v", "\x18": "x", "\x14": "t",
    "\x0e": "n", "\x0c": "l", "\x17": "w", "\x13": "s",
    "\x10": "p", "\x06": "f",
}


class KeyboardMonitor:
    """Collect keyboard behavior metadata without retaining typed characters."""

    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event,
                 config: Config):
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.config = config
        self.operator_active = lambda: False
        self.ignored_keys = set()
        self.pressed: set[object] = set()
        self.pressed_at: dict[object, float] = {}
        self.typing_keys_down: set[object] = set()
        self.last_scored: dict[str, float] = {}
        self.counts: Counter[str] = Counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.lock = threading.Lock()

        self.started_at = time.monotonic()
        self.last_typing_at: float | None = None
        self.total_typing_keys = 0
        self.key_times: deque[float] = deque()
        self.intervals: deque[tuple[float, float]] = deque()
        self.dwells: deque[tuple[float, float]] = deque()
        self.backspace_times: deque[float] = deque()
        self.baseline: dict[str, float] = {}
        self.baseline_ready = False
        self.last_analysis = 0.0
        self.last_metrics_log = 0.0
        self.burst_started_at: float | None = None
        self.idle_before_burst = 0.0
        self.burst_emitted = False
        self.typing_reference = TypingReference(config.typing_reference_kpm)
        self.escape_was_modified = False
        self.escape_generation = 0
        self.hook_failed = False

    @staticmethod
    def _char(key: object) -> str:
        try:
            value = key.char
            if not value:
                return ""
            value = value.lower()
            return CONTROL_CHARS.get(value, value)
        except AttributeError:
            return ""

    def _has(self, *keys: object) -> bool:
        return any(key in self.pressed for key in keys)

    def _ctrl(self) -> bool:
        return self._has(keyboard.Key.ctrl_l, keyboard.Key.ctrl_r)

    def _alt(self) -> bool:
        return self._has(keyboard.Key.alt_l, keyboard.Key.alt_r)

    def _shift(self) -> bool:
        return self._has(keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r)

    def _win(self) -> bool:
        return self._has(keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r)

    def _shortcut(self, key: object, char: str) -> str:
        if self._win() and self._shift() and char == "s":
            return "WIN+SHIFT+S"
        if self._win() and key == keyboard.Key.tab:
            return "WIN+TAB"
        if self._alt() and key == keyboard.Key.tab:
            return "ALT+TAB"
        if self._shift() and key == keyboard.Key.insert:
            return "SHIFT+INSERT"
        if self._ctrl() and char in {"c", "v", "x"}:
            return f"CTRL+{char.upper()}"
        if key == keyboard.Key.print_screen:
            return "PRINT_SCREEN"
        return ""

    def _emit(self, event: RiskEvent, cooldown: float | None = None) -> bool:
        now = event.timestamp
        cooldown = self.config.shortcut_cooldown_sec if cooldown is None else cooldown
        if now - self.last_scored.get(event.kind, 0.0) < cooldown:
            return False
        self.last_scored[event.kind] = now
        self.counts[event.kind] += 1
        self.output_queue.put(event)
        return True

    def _handle_shortcut(self, kind: str, now: float) -> None:
        context = "단축키 관찰; 점수 없음"
        paste_meta = None
        if kind in {"CTRL+V", "SHIFT+INSERT"}:
            paste_meta = clipboard_metadata()
            context = (
                f"붙여넣기; type={paste_meta['type']},chars={paste_meta['chars']},"
                f"lines={paste_meta['lines']}; 내용은 저장하지 않음"
            )
        is_large_paste = bool(
            paste_meta
            and int(paste_meta["chars"]) >= self.config.large_paste_chars
        )
        shortcut_weight = 0.0
        if self._emit(RiskEvent(
            now, "keyboard", kind, shortcut_weight,
            value=kind, context=context,
        )):
            print(f">>> {kind} 감지")

        if is_large_paste:
            chars = int(paste_meta["chars"])
            lines = int(paste_meta["lines"])
            self._emit(RiskEvent(
                now, "clipboard", "LARGE_PASTE",
                self.config.large_paste_weight(chars),
                value=str(chars),
                confidence=min(1.0, 0.5 + chars / 400.0),
                context=f"붙여넣기 단축키 시 클립보드 규모 관찰; chars={chars},lines={lines}; 삽입 성공·외부 출처 미확인; 내용 미저장",
            ))

    def _record_typing_key(self, key: object, now: float) -> None:
        if self._ctrl() or self._alt() or self._win():
            return
        char = self._char(key)
        is_typing = bool(char and ord(char[0]) >= 32) or key in {
            keyboard.Key.space, keyboard.Key.enter, keyboard.Key.backspace,
        }
        if not is_typing:
            return
        with self.lock:
            previous = self.last_typing_at
            idle = now - previous if previous is not None else 0.0
            if previous is not None and idle <= self.config.typing_window_sec:
                self.intervals.append((now, idle))
            if idle >= self.config.effective_long_idle_sec:
                self.burst_started_at = now
                self.idle_before_burst = idle
                self.burst_emitted = False
            self.last_typing_at = now
            self.key_times.append(now)
            self.typing_keys_down.add(key)
            self.total_typing_keys += 1
            self._check_burst(now)
            if key == keyboard.Key.backspace:
                self.backspace_times.append(now)

    def _clear_operator_timing(self):
        # Do not turn time spent writing a proctor decision into an examinee burst.
        with self.lock:
            self.last_typing_at = None
            self.burst_started_at = None
            self.burst_emitted = False
            self.key_times.clear()
            self.intervals.clear()
            self.dwells.clear()
            self.backspace_times.clear()

    def _on_press(self, key: object):
        if self.operator_active():
            self._clear_operator_timing()
            self.ignored_keys.add(key)
            return None
        if key in self.pressed:
            return None
        now = time.monotonic()
        self.pressed.add(key)
        self.pressed_at[key] = now
        if key == keyboard.Key.esc:
            self.escape_generation += 1
            self.escape_was_modified = self._ctrl() or self._alt() or self._shift() or self._win()
            if not self.escape_was_modified:
                # Request exit on press, even if a release event is lost.
                self.stop_event.set()
                return False
        char = self._char(key)
        shortcut = self._shortcut(key, char)
        if shortcut:
            self._handle_shortcut(shortcut, now)
        else:
            self._record_typing_key(key, now)
        return None

    def _on_release(self, key: object):
        if key in self.ignored_keys or self.operator_active():
            self.ignored_keys.discard(key)
            self.pressed.discard(key)
            self.pressed_at.pop(key, None)
            self.typing_keys_down.discard(key)
            return None
        now = time.monotonic()
        started = self.pressed_at.pop(key, None)
        self.pressed.discard(key)
        with self.lock:
            was_typing = key in self.typing_keys_down
            self.typing_keys_down.discard(key)
            if was_typing and started is not None and now >= started:
                self.dwells.append((now, now - started))
        unmodified_escape = (
            key == keyboard.Key.esc
            and not self.escape_was_modified
            and not self._ctrl() and not self._alt()
            and not self._shift() and not self._win()
        )
        if key == keyboard.Key.esc:
            self.escape_was_modified = False
        if unmodified_escape:
            self.stop_event.set()
            print("\nESC 입력 - 프로그램 종료")
            return False
        return None

    @staticmethod
    def _mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    def _snapshot(self, now: float) -> TypingSnapshot:
        cutoff = now - self.config.typing_window_sec
        while self.key_times and self.key_times[0] < cutoff:
            self.key_times.popleft()
        while self.intervals and self.intervals[0][0] < cutoff:
            self.intervals.popleft()
        while self.dwells and self.dwells[0][0] < cutoff:
            self.dwells.popleft()
        while self.backspace_times and self.backspace_times[0] < cutoff:
            self.backspace_times.popleft()

        intervals = [value for _, value in self.intervals]
        dwells = [value for _, value in self.dwells]
        mean_interval = self._mean(intervals)
        interval_cv = (
            statistics.pstdev(intervals) / mean_interval
            if len(intervals) >= 2 and mean_interval > 0 else 0.0
        )
        observed = min(self.config.typing_window_sec, max(1.0, now - self.started_at))
        key_count = len(self.key_times)
        return TypingSnapshot(
            now, key_count, key_count * 60.0 / observed,
            mean_interval, interval_cv, self._mean(dwells),
            len(self.backspace_times) / max(1, key_count),
            now - self.last_typing_at if self.last_typing_at else now - self.started_at,
        )

    def _metric_context(self, snapshot: TypingSnapshot) -> str:
        return (
            f"keys={snapshot.key_count},kpm={snapshot.keys_per_minute:.1f},"
            f"interval={snapshot.mean_interval:.3f},cv={snapshot.interval_cv:.3f},"
            f"dwell={snapshot.mean_dwell:.3f},backspace={snapshot.backspace_ratio:.3f},"
            f"idle={snapshot.idle_sec:.1f}; 입력 내용 미저장"
        )

    def set_typing_reference(self, reference: TypingReference) -> None:
        with self.lock:
            self.typing_reference = reference
            self.baseline = {
                "kpm": reference.reference_kpm,
                "interval_cv": reference.interval_cv or 0.0,
            }
            self.baseline_ready = True
        self.output_queue.put(RiskEvent(
            time.monotonic(), "typing", "TYPING_REFERENCE_READY", 0.0,
            value="PERSONAL_ASSISTED" if reference.valid else "DEFAULT_ONLY",
            context=(f"reference_kpm={reference.reference_kpm:.1f},"
                     f"personal_weight={reference.personal_weight:.3f},"
                     f"keys={reference.key_count},elapsed={reference.elapsed_sec:.1f}; "
                     "잠정 공통 기준, 입력 내용 미저장"),
        ))

    def _check_burst(self, now: float) -> None:
        # Called on each input, so a 3s burst is not missed by the 5s analysis timer.
        if self.burst_started_at is None:
            return
        age = now - self.burst_started_at
        if age > self.config.burst_window_sec:
            self.burst_started_at = None
            return
        keys = sum(t >= self.burst_started_at for t in self.key_times)
        if not self.burst_emitted and keys >= self.config.effective_burst_min_keys:
            self.burst_emitted = True
            self._emit(RiskEvent(
                now, "typing", "LONG_IDLE_THEN_BURST",
                self.config.event_weights["LONG_IDLE_THEN_BURST"],
                value=str(keys), duration=self.idle_before_burst,
                context=f"idle={self.idle_before_burst:.1f}s,burst_keys={keys}",
            ), self.config.typing_event_cooldown_sec)

    def _analyze(self, now: float) -> None:
        if self.operator_active():
            self._clear_operator_timing()
            return
        with self.lock:
            snapshot = self._snapshot(now)

            if now - self.last_metrics_log >= 15.0:
                self.last_metrics_log = now
                self.output_queue.put(RiskEvent(
                    now, "typing", "TYPING_METRICS", 0.0,
                    value=str(snapshot.key_count), context=self._metric_context(snapshot),
                ))

            self._check_burst(now)

            if not self.baseline_ready or snapshot.key_count < 15:
                return
            baseline_kpm = self.baseline["kpm"]
            speed_limit = max(
                self.config.typing_speed_floor_kpm,
                baseline_kpm * self.config.typing_speed_ratio,
            )
            if snapshot.keys_per_minute > speed_limit:
                confidence = min(1.0, snapshot.keys_per_minute / max(speed_limit * 1.5, 1.0))
                self._emit(RiskEvent(
                    now, "typing", "TYPING_SPEED_SPIKE",
                    self.config.event_weights["TYPING_SPEED_SPIKE"],
                    value=f"{snapshot.keys_per_minute:.1f}", confidence=confidence,
                    context=self._metric_context(snapshot),
                ), self.config.typing_event_cooldown_sec)

            baseline_cv = self.baseline["interval_cv"]
            if (
                snapshot.key_count >= 20
                and self.typing_reference.valid
                and snapshot.keys_per_minute > speed_limit
                and baseline_cv >= 0.08
                and snapshot.interval_cv < max(0.04, baseline_cv * 0.35)
            ):
                self._emit(RiskEvent(
                    now, "typing", "TYPING_RHYTHM_ANOMALY",
                    self.config.event_weights["TYPING_RHYTHM_ANOMALY"],
                    value=f"{snapshot.interval_cv:.3f}",
                    context=self._metric_context(snapshot),
                ), self.config.typing_event_cooldown_sec)

    def _run(self) -> None:
        print("키 입력 행동 분석 시작 (입력 내용은 저장하지 않음, 종료: ESC)")
        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        try:
            listener.start()
            while not self.stop_event.wait(0.25):
                if not listener.is_alive():
                    self.hook_failed = True
                    self.output_queue.put(RiskEvent(
                        time.monotonic(), "system", "KEYBOARD_HOOK_STOPPED", 0.0,
                        context="키 입력 감시 중단; ESC 보조 감시와 창 닫기는 유지",
                    ))
                    break
                now = time.monotonic()
                if now - self.last_analysis >= self.config.typing_analysis_interval_sec:
                    self.last_analysis = now
                    self._analyze(now)
        except Exception as error:
            self.hook_failed = True
            self.output_queue.put(RiskEvent(
                time.monotonic(), "system", "KEYBOARD_HOOK_FAILED", 0.0,
                value=type(error).__name__,
                context="키 입력 감시 실패; ESC 보조 감시와 창 닫기는 유지",
            ))
        finally:
            listener.stop()
            try:
                if listener.ident is not None:
                    listener.join(timeout=1.0)
            except Exception:
                self.hook_failed = True

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.ident is not None:
            self.thread.join(timeout=timeout)
