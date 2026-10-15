"""Own only processes started for this exam, never kill by executable name."""
from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from ctypes import wintypes as w
from pathlib import Path


class _StartupInfo(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('lpReserved', w.LPWSTR), ('lpDesktop', w.LPWSTR),
                ('lpTitle', w.LPWSTR), ('dwX', w.DWORD), ('dwY', w.DWORD),
                ('dwXSize', w.DWORD), ('dwYSize', w.DWORD), ('dwXCountChars', w.DWORD),
                ('dwYCountChars', w.DWORD), ('dwFillAttribute', w.DWORD),
                ('dwFlags', w.DWORD), ('wShowWindow', w.WORD), ('cbReserved2', w.WORD),
                ('lpReserved2', ctypes.c_void_p), ('hStdInput', w.HANDLE),
                ('hStdOutput', w.HANDLE), ('hStdError', w.HANDLE)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [('hProcess', w.HANDLE), ('hThread', w.HANDLE),
                ('dwProcessId', w.DWORD), ('dwThreadId', w.DWORD)]


class _BasicLimit(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimit), ('IoInfo', ctypes.c_ulonglong * 6),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [('times', ctypes.c_longlong * 4), ('TotalPageFaultCount', w.DWORD),
                ('TotalProcesses', w.DWORD), ('ActiveProcesses', w.DWORD), ('TotalTerminatedProcesses', w.DWORD)]


class WindowsJob:
    def __init__(self, command: list[str], cwd: Path, console: bool = False, title: str = ''):
        self.k = ctypes.WinDLL('kernel32', use_last_error=True)
        api = {
            'CreateJobObjectW': ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            'SetInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            'QueryInformationJobObject': ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
            'IsProcessInJob': ([w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
            'CreateProcessW': ([w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, w.BOOL,
                                w.DWORD, ctypes.c_void_p, w.LPCWSTR, ctypes.POINTER(_StartupInfo),
                                ctypes.POINTER(_ProcessInfo)], w.BOOL),
            'ResumeThread': ([w.HANDLE], w.DWORD),
            'TerminateProcess': ([w.HANDLE, w.UINT], w.BOOL),
            'TerminateJobObject': ([w.HANDLE, w.UINT], w.BOOL),
            'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            'CloseHandle': ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in api.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result
        self.handle = self.k.CreateJobObjectW(None, None)
        self.process_handle = None
        self.pid = None
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        process = _ProcessInfo()
        try:
            limits = _ExtendedLimit()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            self._check(self.k.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
            startup = _StartupInfo()
            startup.cb = ctypes.sizeof(startup)
            startup.lpTitle = title or 'Capstone exam'
            flags = 0x4 | (0x10 if console else 0x08000000)  # suspended, own console or no console
            self._check(self.k.CreateProcessW(command[0], ctypes.create_unicode_buffer(subprocess.list2cmdline(command)),
                        None, None, False, flags, None, str(cwd), ctypes.byref(startup), ctypes.byref(process)))
            self.process_handle, self.pid = process.hProcess, int(process.dwProcessId)
            # Assign before any code runs, so even early descendants belong to the job.
            self._check(self.k.AssignProcessToJobObject(self.handle, process.hProcess))
            if self.k.ResumeThread(process.hThread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            if process.hProcess:
                self.k.TerminateProcess(process.hProcess, 1)
            self.close()
            raise
        finally:
            if process.hThread: self.k.CloseHandle(process.hThread)

    @staticmethod
    def _check(success):
        if not success: raise ctypes.WinError(ctypes.get_last_error())

    def alive(self) -> bool:
        if not self.handle: return False
        info = _Accounting()
        self._check(self.k.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
        return info.ActiveProcesses > 0

    def owns_pid(self, pid: int) -> bool:
        if not self.handle: return False
        process = self.k.OpenProcess(0x1000, False, pid)
        if not process: return False
        try:
            result = w.BOOL()
            return bool(self.k.IsProcessInJob(process, self.handle, ctypes.byref(result)) and result.value)
        finally:
            self.k.CloseHandle(process)

    def stop(self):
        if self.handle:
            self._check(self.k.TerminateJobObject(self.handle, 0))
            deadline = time.monotonic() + 4
            while self.alive() and time.monotonic() < deadline:
                time.sleep(.05)
            if self.alive(): raise RuntimeError('시험 프로세스 종료 확인 시간이 초과됐습니다.')

    def close(self):
        if self.handle: self.k.CloseHandle(self.handle); self.handle = None
        if self.process_handle: self.k.CloseHandle(self.process_handle); self.process_handle = None


class PosixGroup:
    """Linux backend; full GUI/focus integration still requires deployment testing."""
    def __init__(self, command, cwd, console=False, title=''):
        self.process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        self.pid = self.process.pid
    def alive(self):
        self.process.poll()
        try: os.killpg(self.pid, 0); return True
        except ProcessLookupError: return False
    def owns_pid(self, pid):
        try: return os.getpgid(pid) == self.pid
        except ProcessLookupError: return False
    def stop(self):
        try: os.killpg(self.pid, signal.SIGTERM)
        except ProcessLookupError: return
        deadline = time.monotonic() + 2
        while self.alive() and time.monotonic() < deadline: time.sleep(.05)
        if self.alive():
            try: os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        try: self.process.wait(timeout=2)
        except subprocess.TimeoutExpired: raise RuntimeError('시험 프로세스 종료 실패')
    def close(self): self.stop()


class ExamApplication:
    def __init__(self, profile, session_dir: Path):
        self.profile, self.session_dir = profile, session_dir
        self.process = None
        self.title = 'Capstone exam ' + session_dir.name
    def start(self):
        command, cwd = self.profile.command(self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        backend = WindowsJob if os.name == 'nt' else PosixGroup
        self.process = backend(command, cwd, self.profile.console, self.title)
        return self.process.pid
    def alive(self): return self.process is not None and self.process.alive()
    def stop(self):
        if self.process:
            self.process.stop()
    def close(self):
        if self.process:
            self.process.close()
            self.process = None
    def foreground_allowed(self):
        if os.name != 'nt': return False
        user = ctypes.WinDLL('user32', use_last_error=True)
        user.GetForegroundWindow.restype = w.HWND
        user.GetWindowThreadProcessId.argtypes = [w.HWND, ctypes.POINTER(w.DWORD)]
        user.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, ctypes.c_int]
        hwnd = user.GetForegroundWindow()
        pid = w.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        # Windows console windows may belong to conhost/Terminal, not Python.
        # Their unique per-session title is supplied by CreateProcess startup info.
        title = ctypes.create_unicode_buffer(1024)
        user.GetWindowTextW(hwnd, title, len(title))
        owned = (self.process is not None and self.process.owns_pid(pid.value)) or (
            self.profile.console and self.title in title.value)
        if not owned:
            from activity_monitor import foreground_details, allowed_external_window
            title_text, executable, _ = foreground_details()
            return allowed_external_window(self.profile.allowed_windows, title_text, executable)
        return (not self.profile.window_keywords or
                any(keyword.lower() in title.value.lower() for keyword in self.profile.window_keywords))
