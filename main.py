from __future__ import annotations

import queue
import signal
import sys
import subprocess
import json
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4
import threading
import time
from collections import deque

import cv2

from activity_monitor import ClipboardChangeMonitor, WindowActivityMonitor
from config import Config
from event_logger import EventLogger, print_summary
from exit_monitor import ExitMonitor
from keyboard_monitor import KeyboardMonitor
from models import RiskEvent
from risk_engine import RiskEngine
from review_control import ReviewControl
from review_session import ReviewSession
from ui import WINDOW_TITLE, render
from vision_monitor import VisionMonitor
from typing_setup import run_typing_setup
from exam_profiles import ExamProfile, choose_profile
from exam_process import ExamApplication
from session_lock import SessionLock


def _ignore_console_ctrl_c(_signum, _frame) -> None:
    """Keep Ctrl+C as a monitored shortcut instead of a termination signal."""
    print("\nCtrl+C 감지 - 프로그램은 종료되지 않습니다. 종료는 ESC를 사용하세요.")


def _add_and_log(engine: RiskEngine, logger: EventLogger, event: RiskEvent,
                 recent_events: deque[RiskEvent]) -> None:
    accepted, transition = engine.add(event)
    for item in accepted:
        logger.write(item, engine.score, engine.state)
        if item.weight > 0 or item.kind in {
            "CLIPBOARD_CHANGED", "CALIBRATION_RETRY", "CALIBRATION_VERIFYING"
        }:
            recent_events.append(item)
        if item.weight > 0:
            print(
                f">>> {item.kind}: +{item.weight:.0f} "
                f"| Risk={engine.score:.1f} | {engine.state}"
            )
    if transition:
        logger.write(RiskEvent(
            event.timestamp, "system", "STATE_CHANGE", 0.0,
            value=f"{transition[0]}->{transition[1]}",
            context="현재 위험도 R 단계 변경; 감독관 호출과 독립",
        ), engine.score, engine.state)


def _drain_events(event_queue: queue.Queue, engine: RiskEngine,
                  logger: EventLogger, recent_events: deque[RiskEvent]) -> None:
    while True:
        try:
            _add_and_log(engine, logger, event_queue.get_nowait(), recent_events)
        except queue.Empty:
            return


def _window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        # HighGUI raises when the user has already destroyed the window.
        return True


def _runtime_settings(config: Config) -> str:
    return (
        f"mode={'TEST' if config.test_mode else 'NORMAL'},"
        f"scoring=dual,R_window={config.score_window_sec},R_half_life={config.score_half_life_sec},review_at={config.review_threshold},"
        f"head_left={config.head_left_yaw_threshold:g},"
        f"head_right={config.head_right_yaw_threshold:g},"
        f"head_pitch={config.head_pitch_threshold:g},"
        f"head_hold={config.head_min_duration:g}s,"
        f"head_repeat={config.head_repeat_sec:g}s,"
        f"min_face_size={config.min_face_quality:g}"
    )


def _open_reviewer(log_path: Path, *, control_log_path: Path):
    try:
        return subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("reviewer.py")),
             "--log", str(log_path), "--control-log", str(control_log_path), "--auto-review"],
            cwd=Path(__file__).resolve().parent,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as error:
        print(f"감독관 창 실행 실패: {error}. reviewer.cmd로 다시 열 수 있습니다.")
        return None


def main(profile: ExamProfile | None = None):
    signal.signal(signal.SIGINT, _ignore_console_ctrl_c)
    config = Config()
    if profile:
        config = replace(config, exam_window_keywords=profile.window_keywords or ('capstone exam',))
    runtime_settings = _runtime_settings(config)
    print(f"실행 파일: {__file__}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"적용 설정: {runtime_settings}", flush=True)
    typing_reference = run_typing_setup(config)
    if typing_reference is None:
        return
    session_start = time.monotonic()
    stop_event = threading.Event()
    event_queue: queue.Queue = queue.Queue()
    recent_events: deque[RiskEvent] = deque(maxlen=20)
    session_id = uuid4().hex
    logger = EventLogger(config.log_path, session_start, session_id=session_id)
    review_control = ReviewControl(config.log_path, session_id)
    control_command = None
    end_reason = 'CLOSED'
    engine = RiskEngine(config)
    logger.engine = engine
    review = ReviewSession(engine, logger, review_control, _open_reviewer)
    keyboard_monitor = KeyboardMonitor(event_queue, stop_event, config)
    keyboard_monitor.set_typing_reference(typing_reference)
    window_monitor = WindowActivityMonitor(event_queue, stop_event, config)
    clipboard_monitor = ClipboardChangeMonitor(event_queue, stop_event, config)
    keyboard_monitor.operator_active = review.operator_active
    window_monitor.operator_active = review.operator_active
    exit_monitor = ExitMonitor(stop_event, keyboard_monitor)
    vision_monitor: VisionMonitor | None = None
    cap = None
    target = None
    stop_error = None
    session_dir = config.log_path.parent / 'exam_runs' / session_id

    logger.write(RiskEvent(
        session_start, "system", "SESSION_START", 0.0, value=session_id,
        context=(
            f"{runtime_settings},"
            f"profile={profile.name if profile else 'monitor-only'},"
            f"typing_window={config.typing_window_sec}s,weights={config.event_weights}"
        ),
    ), engine.score, engine.state)

    try:
        vision_monitor = VisionMonitor(config)
        source = config.camera_index if config.use_webcam else config.video_path
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError("영상 또는 웹캠을 열 수 없습니다.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0
        delay = 1 if config.use_webcam else max(1, int(1000 / fps))
        frame_count, last_timestamp_ms = 0, -1

        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        if profile:
            session_dir.mkdir(parents=True, exist_ok=False)
            (session_dir / 'profile.json').write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding='utf-8')
            target = ExamApplication(profile, session_dir)
            pid = target.start()
            logger.write(RiskEvent(time.monotonic(), 'system', 'EXAM_APP_STARTED', 0,
                                  value=profile.name, context=f'pid={pid},session_id={session_id}'), engine.score, engine.state)
            window_monitor.foreground_predicate = target.foreground_allowed
        keyboard_monitor.start()
        window_monitor.start()
        clipboard_monitor.start()
        exit_monitor.start()
        print("정면을 바라보며 얼굴 캘리브레이션을 진행해 주세요.")
        print("타이핑은 공통 기준과 시작 문구 측정을 보조로 사용합니다. 단독 이상은 점수에 반영하지 않습니다.")
        print("점수는 누적됩니다. 미검토 누적점수가 기준에 도달하면 감독관 확인창이 열립니다. 시험과 감시는 계속됩니다.")
        print("종료: 단독 ESC 또는 감시 창의 X 버튼. 감시 종료 시 시험 앱도 닫힙니다.")

        while cap.isOpened() and not stop_event.is_set():
            if target is not None and not target.alive():
                end_reason = 'EXAM_APP_EXITED'
                logger.write(RiskEvent(time.monotonic(), 'system', 'EXAM_APP_EXITED', 0,
                                      context='시험 앱 종료 감지'), engine.score, engine.state)
                break
            command = review_control.poll()
            if command is not None and review.apply(command):
                control_command = command
                end_reason = command.action.upper()
                stop_event.set()
                break
            success, frame = cap.read()
            if not success:
                print("영상 재생 종료")
                break
            if config.use_webcam:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.monotonic()
            timestamp_ms = (
                int((now - session_start) * 1000)
                if config.use_webcam else int(frame_count * 1000 / fps)
            )
            timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
            last_timestamp_ms, frame_count = timestamp_ms, frame_count + 1

            vision = vision_monitor.process(rgb, width, height, timestamp_ms, now)
            engine.set_active_visual(vision.confirmed_active)
            durations = dict(vision.durations)
            focus_since = window_monitor.focus_lost_since
            if window_monitor.focus_lost and focus_since is not None:
                durations["WINDOW_FOCUS_LOST"] = max(0.0, now - focus_since)
            engine.set_sustained(durations)
            _drain_events(event_queue, engine, logger, recent_events)
            for event in vision.events:
                _add_and_log(engine, logger, event, recent_events)

            if vision.calibration_just_completed and vision.calibration_baseline:
                baseline = ",".join(
                    f"{name}={value:.5f}"
                    for name, value in vision.calibration_baseline.items()
                )
                logger.write(RiskEvent(
                    now, "system", "CALIBRATION_COMPLETE", 0.0,
                    value=str(vision.calibration_samples), context=baseline,
                ), engine.score, engine.state)

            transition = engine.refresh(now)
            if transition:
                logger.write(RiskEvent(
                    now, "system", "STATE_CHANGE", 0.0,
                    value=f"{transition[0]}->{transition[1]}",
                    context="현재 위험도 R 단계 변경; 감독관 호출과 독립",
                ), engine.score, engine.state)

            review.tick(now)

            render(
                frame, vision, engine, keyboard_monitor, window_monitor,
                recent_events, now, config,
            )
            cv2.imshow(WINDOW_TITLE, frame)
            key_code = cv2.waitKey(delay) & 0xFF
            if _window_closed():
                stop_event.set()
                break
            # KeyboardMonitor handles unmodified ESC globally. OpenCV's
            # key code alone cannot distinguish ESC from Ctrl+Shift+ESC.
            if key_code in (ord("r"), ord("R")):
                vision_monitor.reset_calibration()
                engine.set_active_visual(set())
                engine.set_sustained({
                    kind: duration for kind, duration in engine.sustained_durations.items()
                    if kind == "WINDOW_FOCUS_LOST"
                })
                logger.write(RiskEvent(
                    time.monotonic(), "system", "CALIBRATION_MANUAL_RESET", 0.0,
                    context="사용자가 R 키로 재캘리브레이션 시작",
                ), engine.score, engine.state)
                print("재캘리브레이션 시작 - 정면을 보고 움직이지 마세요.")
            # q deliberately has no special behavior.
    finally:
        stop_event.set()
        exit_monitor.join()
        keyboard_monitor.join()
        window_monitor.join()
        clipboard_monitor.join()
        if cap is not None:
            cap.release()
        if vision_monitor is not None:
            vision_monitor.close()
        cv2.destroyAllWindows()
        _drain_events(event_queue, engine, logger, recent_events)
        engine.refresh(time.monotonic())
        failed = sys.exc_info()[0] is not None
        review.close()
        # Always stop only this session's owned app, including monitor failure.
        if target is not None:
            try:
                target.stop()
                logger.write(RiskEvent(time.monotonic(), 'system', 'EXAM_APP_STOPPED', 0,
                    value=profile.name, context='reason=' + end_reason), engine.score, engine.state)
            except (OSError, RuntimeError) as error:
                stop_error = str(error)
                logger.write(RiskEvent(time.monotonic(), 'system', 'EXAM_APP_STOP_FAILED', 0,
                    context=str(error)), engine.score, engine.state)
                print(f'시험 앱 종료 확인 실패: {error}', flush=True)
            finally:
                target.close()
        if failed:
            logger.write(RiskEvent(time.monotonic(), 'system', 'SESSION_END', 0,
                                  value='ERROR', context=f'session_id={session_id}'), engine.score, engine.state)
        if failed and control_command is not None:
            review_control.complete(control_command, "error", "모니터링 종료 중 오류가 발생했습니다. 실행 로그를 확인하세요.")
    logger.write(RiskEvent(time.monotonic(), 'system', 'SESSION_END', 0,
                          value=control_command.action.upper() if control_command else end_reason,
                          context=f'session_id={session_id}'), engine.score, engine.state)
    print_summary(engine, keyboard_monitor.counts, config.log_path)
    if control_command is not None:
        if stop_error:
            review_control.complete(control_command, 'error', '시험 앱 종료 확인 실패: ' + stop_error)
            return None
        return review_control, control_command
    return None


def run_application(profile: ExamProfile | None = None) -> None:
    if profile is None:
        profile = choose_profile()
        if profile is None:
            return
    while True:
        result = main(profile=profile)
        if result is None:
            return
        controller, command = result
        if command.action == "stop":
            controller.complete(command, message="모니터링 종료 완료. 감독관 판단과 근거를 보관했습니다.")
            return
        if command.action != 'restart':
            raise ValueError('새 시험은 명시적인 재시험 요청으로만 시작합니다.')
        controller.complete(command, message="기존 세션 보관 및 시험 앱 종료 완료. 새 세션 시작 안내로 이동합니다.")


if __name__ == "__main__":
    try:
        log_path = Config().log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with SessionLock(log_path.with_suffix('.lock')):
            run_application()
    except Exception as error:
        import traceback
        traceback.print_exc()
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror('시험 실행 오류', str(error), parent=root)
        root.destroy()
        raise SystemExit(1)
