import importlib
import os
import subprocess
import sys


def get_git_root(some_dir):
    try:
        return subprocess.check_output(
            'git rev-parse --show-toplevel',
            shell=True, cwd=some_dir
        ).decode(sys.stdout.encoding).strip()
    except subprocess.CalledProcessError:
        return None


def get_git_version(some_dir):
    try:
        return subprocess.check_output(
            'git rev-parse --short HEAD',
            shell=True, cwd=some_dir
        ).decode(sys.stdout.encoding).strip()
    except subprocess.CalledProcessError:
        return None


def get_git_diff(some_dir):
    try:
        return subprocess.check_output(
            'git diff HEAD',
            shell=True, cwd=some_dir
        ).decode(sys.stdout.encoding)
    except subprocess.CalledProcessError:
        return None


def get_module_git_roots():
    root_module_names = set(
        m.split('.')[0] for m in sys.modules
        if not m.startswith('_')
        and m not in sys.builtin_module_names
    )
    root_modules = set(
        (sys.modules[m] if m in sys.modules else importlib.import_module(m))
        for m in root_module_names
    )
    root_dirs = set(
        os.path.dirname(m.__file__) for m in root_modules
        if hasattr(m, '__file__')
        and 'site-packages' not in m.__file__
    )
    git_root_dirs = set(
        get_git_root(d) for d in root_dirs
        if not d.startswith('/usr/lib')
    )
    git_root_dirs = set(
        d for d in git_root_dirs if d is not None
    )
    return git_root_dirs


def get_git_status():
    result = []
    for git_dir in get_module_git_roots():
        head = git_dir + ' @ ' + get_git_version(git_dir)
        diff = get_git_diff(git_dir)
        dir_status = [
            '##################################################################',
            head
        ]
        if len(diff.strip()) > 0:
            dir_status.append('##########')
            dir_status.append(diff)
        result.append('\n'.join(dir_status))

    return '\n\n'.join(result)
