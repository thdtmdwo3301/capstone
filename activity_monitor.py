from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from ctypes import wintypes
from pathlib import Path

from config import Config
from models import RiskEvent


def clipboard_metadata() -> dict[str, int | str]:
    """Read clipboard type and size only. Clipboard contents are never returned."""
    if os.name != "nt":
        return {"type": "UNAVAILABLE", "chars": 0, "lines": 0}

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    cf_unicode_text = 13
    if not user32.IsClipboardFormatAvailable(cf_unicode_text):
        return {"type": "NON_TEXT", "chars": 0, "lines": 0}

    opened = False
    for _ in range(3):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.01)
    if not opened:
        return {"type": "BUSY", "chars": 0, "lines": 0}

    handle = None
    try:
        user32.GetClipboardData.argtypes = [wintypes.UINT]
        user32.GetClipboardData.restype = wintypes.HANDLE
        handle = user32.GetClipboardData(cf_unicode_text)
        if not handle:
            return {"type": "EMPTY", "chars": 0, "lines": 0}
        kernel32.GlobalLock.argtypes = [wintypes.HANDLE]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes = [wintypes.HANDLE]
        kernel32.GlobalUnlock.restype = wintypes.BOOL
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return {"type": "LOCK_FAILED", "chars": 0, "lines": 0}
        try:
            text = ctypes.wstring_at(pointer)
            return {
                "type": "TEXT",
                "chars": len(text),
                "lines": 0 if not text else text.count("\n") + 1,
            }
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def foreground_details() -> tuple[str, str, int]:
    """Return (window title, executable name) without recording window content."""
    if os.name != "nt":
        return "", "UNAVAILABLE", 0
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    # Explicit signatures prevent pointer-sized HWND/HANDLE truncation on x64.
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return "", "UNKNOWN", 0

    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))

    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    process_name = "UNKNOWN"
    process = kernel32.OpenProcess(0x1000, False, process_id.value)
    if process:
        try:
            size = wintypes.DWORD(1024)
            path_buffer = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(
                process, 0, path_buffer, ctypes.byref(size)
            ):
                process_name = path_buffer.value
        finally:
            kernel32.CloseHandle(process)
    return buffer.value, process_name, process_id.value


def _foreground_window() -> tuple[str, str]:
    title, executable, _ = foreground_details()
    return title, Path(executable).name


def allowed_external_window(rules, title, executable):
    # Both exact executable path and a specific title are required. Never store title text.
    return any(os.path.normcase(os.path.abspath(rule['executable'])) == os.path.normcase(os.path.abspath(executable))
               and any(keyword.casefold() in title.casefold() for keyword in rule['title_keywords'])
               for rule in rules)


class WindowActivityMonitor:
    """Detect sustained loss of the configured exam window, including mouse switching."""

    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event,
                 config: Config):
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.config = config
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.armed = False
        self.focus_lost = False
        self.focus_lost_since: float | None = None
        self.current_process = "UNKNOWN"
        self.foreground_predicate = None
        self.operator_active = lambda: False
        self.lost_started = None
        self.last_emit = None
        self.last_sample = None
        self.was_operator = False

    def _is_exam_window(self, title: str) -> bool:
        lowered = title.lower()
        return bool(self.config.exam_window_keywords) and any(
            keyword in lowered for keyword in self.config.exam_window_keywords
        )

    def observe(self, now, process_name, allowed, operator=False):
        if operator or self.was_operator:
            if self.lost_started is not None and self.last_sample is not None:
                self.lost_started += now - self.last_sample
                self.focus_lost_since = self.lost_started
                if self.last_emit is not None:
                    self.last_emit += now - self.last_sample
        self.was_operator = operator
        self.last_sample = now
        if operator:
            return
        if allowed:
            self.armed = True
            if self.lost_started is not None and self.last_emit is not None:
                self.output_queue.put(RiskEvent(now, 'window', 'WINDOW_FOCUS_RESTORED', 0,
                    value='ALLOWED_WINDOW', duration=now-self.lost_started,
                    context=f'허용 창 복귀; process={self.current_process}'))
            self.focus_lost = False
            self.focus_lost_since = self.lost_started = self.last_emit = None
            self.current_process = process_name
            return
        if not self.armed:
            return
        self.focus_lost = True
        self.current_process = process_name
        if self.lost_started is None:
            self.lost_started = now
        self.focus_lost_since = self.lost_started
        duration = now - self.lost_started
        if ((self.last_emit is None and duration >= self.config.focus_lost_min_duration)
                or (self.last_emit is not None and now-self.last_emit >= self.config.focus_lost_repeat_sec)):
            self.last_emit = now
            self.output_queue.put(RiskEvent(now, 'window', 'WINDOW_FOCUS_LOST',
                self.config.event_weights['WINDOW_FOCUS_LOST'], value=process_name, duration=duration,
                context='지속된 비허용 창 이탈; 마우스 전환 포함; 창 제목·입력 내용 미저장'))

    def _run(self) -> None:
        while not self.stop_event.wait(self.config.window_poll_sec):
            title, process_name = _foreground_window()
            allowed = self.foreground_predicate() if self.foreground_predicate else self._is_exam_window(title)
            self.observe(time.monotonic(), process_name, allowed, self.operator_active())

    def start(self) -> None:
        if self.config.exam_window_keywords or self.foreground_predicate:
            self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.ident is not None:
            self.thread.join(timeout=timeout)


class ClipboardChangeMonitor:
    """Observe clipboard sequence changes without retaining clipboard contents."""

    def __init__(self, output_queue: queue.Queue, stop_event: threading.Event,
                 config: Config):
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.config = config
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _sequence_number() -> int:
        if os.name != "nt":
            return 0
        user32 = ctypes.windll.user32
        user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
        return int(user32.GetClipboardSequenceNumber())

    def _run(self) -> None:
        previous = self._sequence_number()
        while not self.stop_event.wait(self.config.clipboard_poll_sec):
            current = self._sequence_number()
            if current == 0 or current == previous:
                continue
            previous = current
            metadata = clipboard_metadata()
            now = time.monotonic()
            self.output_queue.put(RiskEvent(
                now, "clipboard", "CLIPBOARD_CHANGED", 0.0,
                value=f"{metadata['type']}:{metadata['chars']}",
                context=(
                    f"sequence={current},type={metadata['type']},"
                    f"chars={metadata['chars']},lines={metadata['lines']}; 내용 미저장"
                ),
            ))

    def start(self) -> None:
        if os.name == "nt":
            self.thread.start()

    def join(self, timeout: float = 2.0) -> None:
        if self.thread.ident is not None:
            self.thread.join(timeout=timeout)
