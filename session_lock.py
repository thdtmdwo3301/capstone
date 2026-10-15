"""Keep two launchers from mixing one current CSV and command queue."""
import os


class SessionLock:
    def __init__(self, path): self.path, self.stream = path, None
    def __enter__(self):
        self.stream = self.path.open('a+b')
        self.stream.seek(0, 2)
        if self.stream.tell() == 0: self.stream.write(b'0'); self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError('이미 시험 감시가 실행 중입니다. 기존 세션을 종료한 후 다시 시작하세요.')
        return self
    def __exit__(self, *_):
        self.stream.close()
