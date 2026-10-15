import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from buildkite.tests.test_windows_full_suite import prepare_full_runner
from buildkite.windows import run_tests, windows_helpers

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows helpers"
)


def wait_for_file(path):
    deadline = time.monotonic() + 15
    while True:
        if path.is_file():
            content = path.read_text()
            if content:
                return content
        if time.monotonic() >= deadline:
            pytest.fail(f"Child did not create {path}")
        time.sleep(0.01)


@pytest.mark.parametrize("cleanup", ["terminate", "close-handle"])
def test_job_terminates_orphans_without_touching_other_processes(tmp_path, cleanup):
    marker = tmp_path / "child.pid"
    command = (
        "import pathlib, subprocess, sys;"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f"pathlib.Path({str(marker)!r}).write_text(str(p.pid))"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child = windows_helpers.TestProcess(
        {"kind": "command", "command": [sys.executable, "-c", command]}
    )
    descendant_handle = None
    try:
        descendant_pid = int(wait_for_file(marker))
        descendant_handle = child.job.api.OpenProcess(0x00100000, False, descendant_pid)
        assert descendant_handle
        wait = child.job.api.WaitForSingleObject
        wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        wait.restype = ctypes.c_uint32
        assert child.wait(timeout=15) == 0
        assert wait(descendant_handle, 0) == 258  # WAIT_TIMEOUT: orphan is still alive.
        if cleanup == "terminate":
            child.close()
        else:
            child.job.close()
        assert wait(descendant_handle, 5000) == 0
        assert unrelated.poll() is None
        child.close()
    finally:
        if descendant_handle:
            child.job.api.CloseHandle(descendant_handle)
        child.close()
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_job_assignment_failure_never_starts_user_code(tmp_path, monkeypatch):
    marker = tmp_path / "must-not-run"
    original_popen = subprocess.Popen
    processes = []

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    def reject_assignment(self, pid):
        raise PermissionError("Cannot assign the Windows job")

    monkeypatch.setattr(windows_helpers.subprocess, "Popen", popen)
    monkeypatch.setattr(windows_helpers.WindowsJob, "assign", reject_assignment)
    with pytest.raises(PermissionError, match="Cannot assign"):
        windows_helpers.TestProcess(
            {
                "kind": "command",
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
            }
        )
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert not marker.exists()


def helper_source():
    return (
        "def requires_spawn_multiprocessing(): return False\n"
        "def _sanitize_pythonpath_env(env): env['SANITIZED'] = '1'\n"
        "def fork_new_process_for_each_test(f):\n"
        "    raise AssertionError('fork helper was not adapted')\n"
        "spawn_new_process_for_each_test = fork_new_process_for_each_test\n"
        "def create_new_process_for_each_test(method=None):\n"
        "    return fork_new_process_for_each_test if method == 'fork' "
        "else spawn_new_process_for_each_test\n"
        "class RemoteVLLMServer:\n"
        "    def _get_process_termination_timeout(self): return 5\n"
        "    def _terminate_process_tree(self): raise AssertionError('getpgid')\n"
        "class RemoteOpenAIServer(RemoteVLLMServer): pass\n"
        "class RemoteLaunchRenderServer(RemoteVLLMServer): pass\n"
        "class RemoteOpenAIServerCustom(RemoteOpenAIServer):\n"
        "    def _poll(self): return self.proc.exitcode\n"
    )


def test_helper_adapters_propagate_child_outcomes_and_use_real_modules(tmp_path):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    (tests / "utils.py").write_text(helper_source())
    (tests / "state.py").write_text(
        "counter = 0\ndef increment():\n    global counter\n    counter += 1\n"
    )
    (tests / "test_isolation.py").write_text(
        "import os, pytest\n"
        "from tests import state\n"
        "from tests.utils import create_new_process_for_each_test as isolate\n"
        "@pytest.fixture\n"
        "def parent_pid(): return os.getpid()\n"
        "@isolate('fork')\n"
        "@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_pass(parent_pid, value):\n"
        "    assert os.getpid() != parent_pid\n"
        "    assert state.counter == 0\n"
        "    state.increment()\n"
        "    assert state.counter == 1\n"
        "    assert value in (1, 2)\n"
        "@isolate('fork')\n"
        "def test_fail(): raise ValueError('failure from child')\n"
        "@isolate('spawn')\n"
        "def test_skip(): pytest.skip('skip from child')\n"
        "@isolate()\n"
        "def test_xfail(): pytest.xfail('xfail from child')\n"
        "@isolate('fork')\n"
        "def test_crash(): os._exit(23)\n"
        "@isolate('fork')\n"
        "def test_exit_without_result(): os._exit(0)\n"
        "def test_parent_unchanged(): assert state.counter == 0\n"
    )
    (tests / "unpackaged").mkdir()
    (tests / "unpackaged" / "test_standalone.py").write_text(
        "from tests.utils import create_new_process_for_each_test\n"
        "@create_new_process_for_each_test('fork')\n"
        "def test_standalone(): assert True\n"
    )
    result = subprocess.run(
        [
            str(python),
            "-I",
            str(Path(run_tests.__file__)),
            "--source-dir",
            str(source),
            "--output",
            str(output),
        ],
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    summary = json.loads((output / "pytest-summary.json").read_text())
    assert summary["collected"] == summary["completed"] == 9
    assert summary["passed"] == 4
    assert summary["failed"] == 3
    assert summary["skipped"] == 2
    assert summary["windows_test_helpers"] == [
        "tests.utils.process_isolation",
        "tests.utils.server_process_tree",
    ]
    assert "failure from child" in result.stdout
    assert "ValueError" in result.stdout
    assert "exited with code 23" in result.stdout
    assert "exited with code 0" in result.stdout
    assert "skip from child" in result.stdout
    assert "xfail from child" in result.stdout
    assert "WinError 32" not in result.stdout


@pytest.mark.parametrize(
    "name,prefix",
    [
        ("RemoteOpenAIServer", ["serve"]),
        ("RemoteLaunchRenderServer", ["launch", "render"]),
        ("RemoteOpenAIServerCustom", None),
    ],
)
def test_server_launchers_keep_arguments_and_sanitized_environment(
    monkeypatch, name, prefix
):
    module = SimpleNamespace()
    exec(helper_source(), module.__dict__)
    windows_helpers.adapt_helpers(module)
    process = Mock(spec=windows_helpers.TestProcess)
    factory = Mock(return_value=process)
    monkeypatch.setattr(windows_helpers, "TestProcess", factory)
    server = getattr(module, name)()
    server.child_process_fxn = lambda *args: None
    server._start_server("model", ["--port", "1234"], {"TEST_OVERRIDE": "ok"})
    task, env = factory.call_args.args
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert env["TEST_OVERRIDE"] == "ok"
    assert env["SANITIZED"] == "1"
    if prefix:
        assert task["command"] == ["vllm", *prefix, "model", "--port", "1234"]
    else:
        assert task["function"] is server.child_process_fxn
        assert task["args"] == ({"TEST_OVERRIDE": "ok"}, "model", ["--port", "1234"])
    assert server.proc is process


def test_unknown_helper_layout_fails_instead_of_silently_using_posix():
    with pytest.raises(RuntimeError, match="Unsupported fork test helper layout"):
        windows_helpers.adapt_helpers(SimpleNamespace())


def test_custom_server_is_terminated_as_an_owned_job(tmp_path):
    marker = tmp_path / "custom-started"

    def server_function(env, model, args):
        Path(model).write_text(env["MESSAGE"])
        time.sleep(60)

    module = SimpleNamespace()
    exec(helper_source(), module.__dict__)
    windows_helpers.adapt_helpers(module)
    server = module.RemoteOpenAIServerCustom()
    server.child_process_fxn = server_function
    server._start_server(str(marker), [], {"MESSAGE": "ready"})
    try:
        assert wait_for_file(marker) == "ready"
        assert server._poll() is None
        server._terminate_process_tree()
        assert server._poll() is not None
        server._terminate_process_tree()
    finally:
        server.proc.close()
