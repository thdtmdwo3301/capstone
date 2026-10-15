"""User-editable launch profiles shared by coding and standalone CBT exams."""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / 'exam_profiles.json'


@dataclass(frozen=True)
class ExamProfile:
    id: str
    name: str
    executable: str
    arguments: tuple[str, ...] = ()
    working_directory: str = ''
    kind: str = 'generic'
    platform: str = 'windows'
    console: bool = False
    window_keywords: tuple[str, ...] = ()
    allowed_windows: tuple[dict, ...] = ()

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError('프로필은 항목별 객체여야 합니다.')
        for name in ('arguments', 'window_keywords'):
            if not isinstance(data.get(name, []), list) or not all(isinstance(v, str) for v in data.get(name, [])):
                raise ValueError(f'{name}은 문자열 목록이어야 합니다.')
        for name in ('id', 'name', 'executable', 'working_directory', 'kind', 'platform'):
            if name in data and not isinstance(data[name], str):
                raise ValueError(f'{name}은 문자열이어야 합니다.')
        if not isinstance(data.get('console', False), bool):
            raise ValueError('console은 true/false여야 합니다.')
        rules = data.get('allowed_windows', [])
        if not isinstance(rules, list):
            raise ValueError('allowed_windows는 규칙 목록이어야 합니다.')
        for rule in rules:
            if (not isinstance(rule, dict) or set(rule) != {'executable', 'title_keywords'}
                    or not isinstance(rule['executable'], str) or not Path(rule['executable']).is_absolute()
                    or not isinstance(rule['title_keywords'], list) or not rule['title_keywords']
                    or not all(isinstance(k, str) and k.strip() for k in rule['title_keywords'])):
                raise ValueError('허용 창 규칙에는 실행 파일 절대경로와 비어 있지 않은 제목 키워드 목록이 필요합니다.')
        result = cls(**{**data, 'arguments': tuple(data.get('arguments', [])),
                        'window_keywords': tuple(data.get('window_keywords', [])), 'allowed_windows': tuple(rules)})
        if not result.id or not result.name.strip() or not result.executable.strip():
            raise ValueError('프로필 ID, 이름, 실행 파일을 입력하세요.')
        if result.kind not in {'generic', 'vscode', 'ipython'} or result.platform not in {'windows', 'linux'}:
            raise ValueError('지원하지 않는 실행 방식 또는 운영체제입니다.')
        return result

    def expand(self, text: str, session_dir: Path) -> str:
        for key, value in {'python': sys.executable, 'project': str(BASE_DIR),
                           'session_dir': str(session_dir)}.items():
            text = text.replace('{' + key + '}', value)
        return os.path.expanduser(os.path.expandvars(text))

    def command(self, session_dir: Path) -> tuple[list[str], Path]:
        current = 'windows' if os.name == 'nt' else 'linux'
        if self.platform != current:
            raise ValueError(f'{self.name}: {self.platform}용 프로필입니다. 현재 PC는 {current}입니다.')
        executable = self.expand(self.executable, session_dir)
        resolved = shutil.which(executable) or (str(Path(executable).resolve()) if Path(executable).is_file() else '')
        if not resolved:
            raise ValueError(f'실행 파일을 찾을 수 없습니다: {executable}')
        if os.name == 'nt' and Path(resolved).suffix.lower() != '.exe':
            raise ValueError('Windows에서는 실제 .exe 파일을 선택하세요. .cmd/.bat 바로가기는 지원하지 않습니다.')
        if Path(resolved).name.lower() in {'wsl.exe', 'ubuntu.exe', 'wsl', 'ubuntu'}:
            raise ValueError('WSL 실행기로 원격 Linux 앱을 제어할 수 없습니다. Ubuntu 안에서 감시기를 실행하고 앱을 등록하세요.')
        cwd = Path(self.expand(self.working_directory or '{project}', session_dir)).resolve()
        if not cwd.is_dir():
            raise ValueError(f'작업 폴더를 찾을 수 없습니다: {cwd}')
        args = [self.expand(value, session_dir) for value in self.arguments]
        if self.kind == 'vscode':
            if Path(resolved).with_name('data').is_dir():
                raise ValueError('Portable VS Code는 독립 프로필이 무시됩니다. 일반 설치본 Code.exe를 선택하세요.')
            if any(a.startswith(('--user-data-dir', '--reuse-window')) for a in args):
                raise ValueError('VS Code 독립 실행 옵션은 자동으로 지정됩니다. 중복 옵션을 제거하세요.')
            args = ['--new-window', '--user-data-dir', str(session_dir / 'vscode-profile'), *args]
        return [resolved, *args], cwd


def default_profiles():
    if os.name == 'nt':
        candidates = [Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs/Microsoft VS Code/Code.exe',
                      Path(os.environ.get('ProgramFiles', '')) / 'Microsoft VS Code/Code.exe']
        code = next((str(path) for path in candidates if path.is_file()), 'Code.exe')
    else:
        code = 'code'
    platform = 'windows' if os.name == 'nt' else 'linux'
    return [ExamProfile('vscode', '코드 시험 · VS Code', code, ('{project}/exam_workspace',),
                        '{project}/exam_workspace', 'vscode', platform),
            ExamProfile('ipython', '코드 시험 · IPython', '{python}', ('-m', 'IPython'),
                        '{project}/exam_workspace', 'ipython', platform, True)]


def load_profiles(path: Path = PROFILE_PATH) -> list[ExamProfile]:
    if not path.exists():
        return default_profiles()
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('profiles'), list):
        raise ValueError('시험 프로필 파일 형식이 잘못됐습니다.')
    profiles = [ExamProfile.from_dict(row) for row in data['profiles']]
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError('중복된 프로필 ID가 있습니다.')
    return profiles


def save_profiles(profiles: list[ExamProfile], path: Path = PROFILE_PATH):
    if len({p.id for p in profiles}) != len(profiles):
        raise ValueError('중복된 프로필 ID가 있습니다.')
    payload = {'version': 1, 'profiles': [asdict(profile) for profile in profiles]}
    # Validate the serialized form before replacing an existing configuration.
    for row in json.loads(json.dumps(payload))['profiles']:
        ExamProfile.from_dict(row)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def choose_profile() -> ExamProfile | None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    root = tk.Tk()
    root.title('시험 환경 선택 · 캡스톤')
    root.geometry('900x770')
    root.minsize(780, 700)
    panel = ttk.Frame(root, padding=20)
    panel.pack(fill='both', expand=True)
    ttk.Label(panel, text='시험 환경 선택', font=('Malgun Gothic', 18, 'bold')).pack(anchor='w')
    ttk.Label(panel, text='코드 시험·CBT 등 시험 프로그램의 실행 설정을 추가하고 선택하세요.').pack(anchor='w', pady=8)
    try:
        profiles = load_profiles()
    except (OSError, ValueError, TypeError) as error:
        messagebox.showerror('시험 설정 읽기 실패', str(error), parent=root)
        root.destroy()
        return None
    (BASE_DIR / 'exam_workspace').mkdir(exist_ok=True)
    picker = ttk.Combobox(panel, state='readonly', values=[p.name for p in profiles])
    picker.pack(fill='x')
    form = ttk.Frame(panel)
    form.pack(fill='both', expand=True, pady=12)
    form.columnconfigure(1, weight=1)
    name, executable, cwd = tk.StringVar(), tk.StringVar(), tk.StringVar()
    kind_labels = {'generic': '일반 프로그램 · CBT', 'vscode': 'VS Code 독립 시험 창', 'ipython': 'IPython 콘솔'}
    kind, platform = tk.StringVar(value=kind_labels['generic']), tk.StringVar(value='windows' if os.name == 'nt' else 'linux')
    console = tk.BooleanVar()
    fields = [('환경 이름', name), ('실행 파일', executable), ('작업 폴더', cwd)]
    for row, (label, variable) in enumerate(fields):
        ttk.Label(form, text=label).grid(row=row, column=0, sticky='w', padx=(0, 12), pady=6)
        ttk.Entry(form, textvariable=variable).grid(row=row, column=1, sticky='ew')
    def pick_exe():
        value = filedialog.askopenfilename(parent=root, title='시험 프로그램 실행 파일')
        if value: executable.set(value)
    def pick_dir():
        value = filedialog.askdirectory(parent=root, title='시험 작업 폴더')
        if value: cwd.set(value)
    ttk.Button(form, text='찾기', command=pick_exe).grid(row=1, column=2, padx=5)
    ttk.Button(form, text='찾기', command=pick_dir).grid(row=2, column=2, padx=5)
    ttk.Label(form, text='운영체제').grid(row=3, column=0, sticky='w', pady=6)
    ttk.Combobox(form, textvariable=platform, values=('windows', 'linux'), state='readonly').grid(row=3, column=1, sticky='w')
    ttk.Label(form, text='실행 방식').grid(row=4, column=0, sticky='w', pady=6)
    ttk.Combobox(form, textvariable=kind, values=tuple(kind_labels.values()), state='readonly', width=26).grid(row=4, column=1, sticky='w')
    ttk.Label(form, text='Ubuntu용 설정도 보관할 수 있으며, 실행 검증은 해당 PC에서 필요합니다.').grid(row=5, column=1, sticky='w')
    ttk.Label(form, text='실행 옵션\n한 줄에 하나').grid(row=6, column=0, sticky='nw', pady=10)
    args = tk.Text(form, height=4, width=50, font=('Malgun Gothic', 10))
    args.grid(row=6, column=1, columnspan=2, sticky='ew', pady=10)
    ttk.Checkbutton(form, text='별도 콘솔 창 열기 (IPython 등)', variable=console).grid(row=7, column=1, sticky='w')
    ttk.Label(form, text='창 제목 키워드\n선택·쉼표 구분').grid(row=8, column=0, sticky='w', pady=10)
    keywords = tk.StringVar()
    ttk.Entry(form, textvariable=keywords).grid(row=8, column=1, sticky='ew')
    ttk.Label(form, text='실행한 앱의 창 제한용입니다. 외부 문제 페이지·터미널은 exam_profiles.json의 allowed_windows로 지정합니다.', wraplength=610).grid(row=9, column=1, sticky='w')
    ttk.Label(panel, text='사건 점수 누적 → 기준 도달 시 감독관 호출, 시험과 감시는 계속\n'
              '이 실행기로 시작한 시험 앱은 감시 종료 시에도 닫힙니다. 답안은 수시로 저장하세요.\n'
              '기존 앱에 연결하는 실행기·웹 CBT·원격 환경은 별도 연동 검증이 필요합니다.',
              wraplength=810).pack(anchor='w', pady=10)
    current_id = None
    result = None
    note = tk.StringVar(value='프로필은 이 PC의 exam_profiles.json에 저장됩니다.')
    def load(_=None):
        nonlocal current_id
        index = picker.current()
        if index < 0: return
        p = profiles[index]
        current_id = p.id
        for variable, value in ((name, p.name), (executable, p.executable), (cwd, p.working_directory),
                                (kind, kind_labels[p.kind]), (platform, p.platform), (console, p.console),
                                (keywords, ','.join(p.window_keywords))): variable.set(value)
        args.delete('1.0', 'end')
        args.insert('1.0', '\n'.join(p.arguments))
    def new():
        nonlocal current_id
        current_id = None
        picker.set('새 환경')
        name.set('새 시험 환경'); executable.set(''); cwd.set(str(BASE_DIR / 'exam_workspace'))
        kind.set(kind_labels['generic']); console.set(False); keywords.set(''); args.delete('1.0', 'end')
    def save(start=False):
        nonlocal result, current_id
        try:
            p = ExamProfile.from_dict({'id': current_id or uuid4().hex, 'name': name.get().strip(),
                'executable': executable.get().strip(), 'working_directory': cwd.get().strip(),
                'kind': next(key for key, label in kind_labels.items() if label == kind.get()), 'platform': platform.get(), 'console': console.get(),
                'arguments': [s for s in args.get('1.0', 'end-1c').splitlines() if s],
                'allowed_windows': list(next((old.allowed_windows for old in profiles if old.id == current_id), ())),
                'window_keywords': [s.strip().lower() for s in keywords.get().split(',') if s.strip()]})
            if start:
                p.command(BASE_DIR / 'exam_runs' / 'validation')
                if p.kind == 'ipython':
                    import subprocess
                    command, folder = p.command(BASE_DIR / 'exam_runs' / 'validation')
                    try:
                        check = subprocess.run([command[0], '-c', 'import IPython'], cwd=folder,
                            capture_output=True, timeout=10, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    except subprocess.TimeoutExpired:
                        raise ValueError('선택한 Python의 IPython 확인 시간이 초과됐습니다.')
                    if check.returncode:
                        raise ValueError('선택한 Python에 IPython이 설치되어 있지 않습니다. 실행 환경을 확인하세요.')
            updated = [p if old.id == p.id else old for old in profiles]
            if current_id is None: updated.append(p)
            save_profiles(updated)
            profiles[:] = updated
            current_id = p.id
            picker.configure(values=[old.name for old in profiles])
            picker.current(next(i for i, old in enumerate(profiles) if old.id == p.id))
            note.set('환경 설정을 저장했습니다.')
            if start:
                result = p
                root.destroy()
        except (OSError, ValueError, TypeError) as error:
            messagebox.showerror('환경 설정 확인', str(error), parent=root)
    picker.bind('<<ComboboxSelected>>', load)
    buttons = ttk.Frame(panel)
    buttons.pack(fill='x')
    ttk.Button(buttons, text='환경 추가', command=new).pack(side='left')
    ttk.Button(buttons, text='설정 저장', command=save).pack(side='left', padx=8)
    ttk.Button(buttons, text='이 환경으로 시험 시작', command=lambda: save(True)).pack(side='right')
    ttk.Label(panel, textvariable=note).pack(anchor='w', pady=8)
    if profiles: picker.current(0); load()
    else: new()
    root.mainloop()
    return result
