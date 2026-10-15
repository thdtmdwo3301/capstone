"""Local, controlled typing sample. Text is checked in memory, never logged."""
from __future__ import annotations

import statistics
import time
import unicodedata
from dataclasses import dataclass

from config import Config


PHRASE = "본인은 시험 중 타인의 개입이나 허용되지 않은 기기 및 프로그램의 도움을 받지 않겠습니다."


@dataclass(frozen=True)
class TypingReference:
    reference_kpm: float
    personal_kpm: float | None = None
    personal_weight: float = 0.0
    interval_cv: float | None = None
    key_count: int = 0
    elapsed_sec: float = 0.0
    valid: bool = False


def make_reference(config: Config, times: list[float], elapsed: float,
                   pasted: bool = False) -> TypingReference:
    gaps = [b - a for a, b in zip(times, times[1:]) if 0.015 <= b - a <= 1.0]
    valid = (not pasted and len(times) >= 40 and len(gaps) >= 25
             and elapsed >= 6.0)
    if not valid:
        return TypingReference(config.typing_reference_kpm, key_count=len(times),
                               elapsed_sec=elapsed)
    # Active key rate excludes long reading/IME pauses. This is physical keys/min,
    # not Korean characters/min or a population-average typing speed.
    mean = statistics.fmean(gaps)
    personal = 60.0 / mean
    cv = statistics.pstdev(gaps) / mean
    weight = config.typing_personal_max_weight * min(1.0, len(gaps) / 100.0)
    bounded = max(config.typing_reference_kpm * 0.75,
                  min(config.typing_reference_kpm * 1.25, personal))
    reference = ((1.0 - weight) * config.typing_reference_kpm + weight * bounded)
    return TypingReference(reference, personal, weight, cv, len(times), elapsed, True)


def run_typing_setup(config: Config) -> TypingReference | None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("시험 시작 안내 및 입력 확인")
    root.geometry("760x460")
    root.minsize(660, 430)
    root.attributes("-topmost", True)
    result: TypingReference | None = None
    times: list[float] = []
    started: float | None = None
    pasted = False
    note = tk.StringVar(value="아래 문장을 평소 속도로 직접 입력해 주세요.")
    panel = ttk.Frame(root, padding=24)
    panel.pack(fill="both", expand=True)
    ttk.Label(panel, text="시험 시작 전 확인", font=("Malgun Gothic", 16, "bold")).pack(anchor="w")
    ttk.Label(panel, text=PHRASE, wraplength=690, font=("Malgun Gothic", 12)).pack(anchor="w", pady=18)
    ttk.Label(panel, text="입력 내용은 저장하지 않습니다. 측정한 시간과 입력 횟수만 사용합니다.",
              wraplength=690).pack(anchor="w")
    entry = tk.Text(panel, height=4, wrap="word", font=("Malgun Gothic", 12), undo=False)
    entry.pack(fill="x", pady=12)
    ttk.Label(panel, textvariable=note, wraplength=690).pack(anchor="w", pady=8)

    def on_key(event):
        nonlocal started
        if event.state & 0x4 or event.keysym in {"Shift_L", "Shift_R", "Control_L", "Control_R",
                                               "Alt_L", "Alt_R", "Escape", "Return", "Tab"}:
            return
        # Korean IME may send empty char/unknown keysym for a physical key.
        # Windows letter/digit VK codes still identify the press without text.
        physical = 48 <= event.keycode <= 90 or event.keycode in {8, 32, 186, 187, 188, 189, 190, 191, 192, 219, 220, 221, 222}
        if event.char or physical or event.keysym in {"BackSpace", "space"}:
            now = time.monotonic()
            if started is None:
                started = now
            times.append(now)

    def block_paste(_event=None):
        nonlocal pasted
        pasted = True
        note.set("붙여넣기는 기준 측정에 사용할 수 없습니다. 다시 입력을 눌러 직접 입력해 주세요.")
        return "break"

    def reset():
        nonlocal started, pasted
        entry.delete("1.0", "end")
        times.clear()
        started, pasted = None, False
        note.set("아래 문장을 평소 속도로 직접 입력해 주세요.")
        entry.focus_set()

    def finish():
        nonlocal result
        text = unicodedata.normalize("NFC", entry.get("1.0", "end-1c")).strip()
        if text != PHRASE:
            note.set("안내 문장과 입력한 문장이 다릅니다. 띄어쓰기와 마침표를 확인해 주세요.")
            return
        elapsed = time.monotonic() - started if started is not None else 0.0
        candidate = make_reference(config, times, elapsed, pasted)
        if not candidate.valid:
            note.set("측정할 입력이 부족하거나 붙여넣기가 감지됐습니다. 다시 입력하거나 기본 기준으로 시작해 주세요.")
            return
        result = candidate
        root.destroy()

    def fallback():
        nonlocal result
        result = TypingReference(config.typing_reference_kpm)
        root.destroy()

    buttons = ttk.Frame(panel)
    buttons.pack(fill="x", pady=12)
    ttk.Button(buttons, text="입력 확인 후 시작", command=finish).pack(side="left")
    ttk.Button(buttons, text="다시 입력", command=reset).pack(side="left", padx=8)
    ttk.Button(buttons, text="측정 생략 · 기본 기준으로 시작", command=fallback).pack(side="right")
    entry.bind("<KeyPress>", on_key)
    for binding in ("<<Paste>>", "<Control-v>", "<Control-V>", "<Shift-Insert>"):
        entry.bind(binding, block_paste)
    root.bind("<Escape>", lambda _event: root.destroy())
    entry.focus_set()
    root.mainloop()
    return result
