"""Independent Windows exit-key polling, separate from the keyboard hook."""
from __future__ import annotations

import ctypes
import os
import threading


def key_down(virtual_key: int) -> bool:
    if os.name != "nt":
        return False
    query = ctypes.windll.user32.GetAsyncKeyState
    query.argtypes = [ctypes.c_int]
    query.restype = ctypes.c_short
    # Only the current physical state; do not use the shared 'pressed since' bit.
    return bool(query(virtual_key) & 0x8000)


class ExitMonitor:
    def __init__(self, stop_event: threading.Event, keyboard_monitor):
        self.stop_event = stop_event
        self.keyboard_monitor = keyboard_monitor
        self.escape_down = False
        self.released_generation = keyboard_monitor.escape_generation
        self.thread = threading.Thread(target=self._run, daemon=True)

    def poll(self) -> None:
        if self.keyboard_monitor.operator_active():
            self.escape_down = key_down(0x1B)
            return
        down = key_down(0x1B)
        if not down:
            # Remember old hook events without altering the hook's press/release
            # pairing: physical release can be observed before its callback.
            self.released_generation = self.keyboard_monitor.escape_generation
        if down and not self.escape_down:
            modified = any(key_down(vk) for vk in (0x10, 0x11, 0x12, 0x5B, 0x5C))
            hook_modified = (
                self.keyboard_monitor.escape_generation > self.released_generation
                and self.keyboard_monitor.escape_was_modified
            )
            if not modified and not hook_modified:
                self.stop_event.set()
        self.escape_down = down

    def _run(self) -> None:
        while not self.stop_event.wait(0.025):
            self.poll()

    def start(self) -> None:
        if os.name == "nt":
            self.thread.start()

    def join(self, timeout: float = 1.0) -> None:
        if self.thread.ident is not None:
            self.thread.join(timeout)
