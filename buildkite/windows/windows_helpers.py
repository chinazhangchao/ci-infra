"""Windows adapters for the pinned fork's test process helpers, not os emulation."""

import ctypes
import functools
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import weakref
from importlib.machinery import PathFinder
from pathlib import Path

SOURCE_ENV = "_VLLM_CI_TEST_SOURCE"
CHILD_ENV = "_VLLM_CI_TEST_CHILD"
APPLIED_HELPERS = []


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", ctypes.c_uint32),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("active_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority", ctypes.c_uint32),
        ("scheduling", ctypes.c_uint32),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("times", ctypes.c_int64 * 4),
        ("page_faults", ctypes.c_uint32),
        ("total_processes", ctypes.c_uint32),
        ("active_processes", ctypes.c_uint32),
        ("terminated_processes", ctypes.c_uint32),
    ]


class WindowsJob:
    def __init__(self):
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": (ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_wchar_p]),
            "SetInformationJobObject": (
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
            ),
            "QueryInformationJobObject": (
                ctypes.c_int,
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.c_void_p,
                ],
            ),
            "OpenProcess": (
                ctypes.c_void_p,
                [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32],
            ),
            "AssignProcessToJobObject": (
                ctypes.c_int,
                [ctypes.c_void_p, ctypes.c_void_p],
            ),
            "TerminateJobObject": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint32]),
            "CloseHandle": (ctypes.c_int, [ctypes.c_void_p]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.api, name)
            function.restype, function.argtypes = result, args
        self.handle = self.api.CreateJobObjectW(None, None)
        self._check(self.handle)
        self.finalizer = weakref.finalize(self, self.api.CloseHandle, self.handle)
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self._check(
                self.api.SetInformationJobObject(
                    self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                )
            )
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, pid):
        # The child is blocked on stdin until it has joined this job.
        process = self.api.OpenProcess(0x0100 | 0x0001, False, pid)
        self._check(process)
        try:
            self._check(self.api.AssignProcessToJobObject(self.handle, process))
        finally:
            self._check(self.api.CloseHandle(process))

    def terminate(self, timeout):
        if not self.handle:
            return
        self._check(self.api.TerminateJobObject(self.handle, 1))
        deadline = time.monotonic() + timeout
        while True:
            info = _Accounting()
            self._check(
                self.api.QueryInformationJobObject(
                    self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None
                )
            )
            if info.active_processes == 0:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("Windows test process tree did not terminate.")
            time.sleep(0.01)

    def close(self):
        if self.handle:
            self._check(self.api.CloseHandle(self.handle))
            self.handle = None
            self.finalizer.detach()


class TestProcess:
    def __init__(self, task, env=None):
        import cloudpickle

        # Match spawn's import-path inheritance before unpickling fixture arguments.
        paths = [os.getcwd() if path == "" else path for path in sys.path]
        payload = json.dumps(paths).encode("utf-8") + b"\n" + cloudpickle.dumps(task)
        self.job = WindowsJob()
        self.process = None
        try:
            self.process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--child"],
                stdin=subprocess.PIPE,
                env=env,
            )
            self.job.assign(self.process.pid)
            self.process.stdin.write(payload)
            self.process.stdin.close()
            self.process.stdin = None
        except BaseException:
            try:
                if self.process is not None:
                    self.process.kill()
                    self.process.wait()
                    if self.process.stdin is not None:
                        self.process.stdin.close()
            finally:
                self.job.close()
            raise

    @property
    def pid(self):
        return self.process.pid

    @property
    def exitcode(self):
        return self.process.poll()

    @property
    def returncode(self):
        return self.process.returncode

    def poll(self):
        return self.process.poll()

    def wait(self, timeout=None):
        return self.process.wait(timeout=timeout)

    def is_alive(self):
        return self.poll() is None

    def join(self, timeout=None):
        try:
            self.wait(timeout)
        except subprocess.TimeoutExpired:
            return

    def terminate(self):
        self.close()

    kill = terminate

    def close(self, timeout=15):
        try:
            self.job.terminate(timeout)
            self.process.wait(timeout=timeout)
        finally:
            self.job.close()


def _run_child():
    import cloudpickle

    sys.path[:] = json.loads(sys.stdin.buffer.readline())
    task = cloudpickle.loads(sys.stdin.buffer.read())
    if task["kind"] == "command":
        return subprocess.call(task["command"], stdin=subprocess.DEVNULL)
    if task["kind"] == "server":
        task["function"](*task["args"])
        return 0
    if task["kind"] != "test":
        raise ValueError(f"Unsupported Windows child task: {task['kind']}")

    from _pytest.outcomes import Skipped, XFailed

    try:
        target = importlib.import_module(task["module"])
        for name in task["qualname"].split("."):
            target = getattr(target, name)
        target(*task["args"], **task["kwargs"])
        result = {"outcome": "passed"}
    except Skipped as error:
        result = {"outcome": "skipped", "reason": str(error)}
    except XFailed as error:
        result = {"outcome": "xfailed", "reason": str(error)}
    except BaseException as error:
        result = {
            "outcome": "failed",
            "traceback": traceback.format_exc(),
            "exception": error if isinstance(error, Exception) else None,
        }
        try:
            cloudpickle.dumps(result)
        except Exception as serialization_error:
            print(
                f"Could not serialize child exception: {serialization_error}; "
                "reporting its traceback instead.",
                file=sys.stderr,
            )
            result["exception"] = None
    with open(task["result"], "wb") as stream:
        cloudpickle.dump(result, stream)
    return 1 if result["outcome"] == "failed" else 0


def spawn_test(function):
    identity = f"{function.__module__}.{function.__qualname__}"
    if "<locals>" in function.__qualname__:
        raise ValueError(f"Windows isolated tests must be importable: {identity}")

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if os.environ.get(CHILD_ENV) == identity:
            return function(*args, **kwargs)
        import cloudpickle
        import pytest

        with tempfile.TemporaryDirectory(prefix="vllm-test-process-") as temporary:
            result_path = Path(temporary) / "result.pkl"
            task = {
                "kind": "test",
                "module": function.__module__,
                "qualname": function.__qualname__,
                "args": args,
                "kwargs": kwargs,
                "result": str(result_path),
            }
            child = TestProcess(task, {**os.environ, CHILD_ENV: identity})
            try:
                returncode = child.wait()
            finally:
                child.close()
            if not result_path.is_file():
                raise RuntimeError(
                    f"Test subprocess {identity} exited with code {returncode} "
                    "without reporting an outcome."
                )
            with result_path.open("rb") as stream:
                result = cloudpickle.load(stream)
            outcome = result["outcome"]
            if outcome == "failed":
                error = result.get("exception")
                if isinstance(error, Exception):
                    error.add_note(result["traceback"])
                    raise error
                raise RuntimeError(result["traceback"])
            if returncode:
                raise RuntimeError(
                    f"Test subprocess {identity} exited with code {returncode}."
                )
            if outcome == "skipped":
                pytest.skip(result["reason"])
            if outcome == "xfailed":
                pytest.xfail(result["reason"])
            if outcome != "passed":
                raise RuntimeError(f"Unknown child test outcome: {outcome}")

    return wrapper


def _terminate_server(self):
    if not isinstance(self.proc, TestProcess):
        raise TypeError("Windows server was not launched in an owned test job.")
    self.proc.close(timeout=self._get_process_termination_timeout())


def _server_launcher(module, prefix):
    def start(self, model, vllm_serve_args, env_dict):
        env = {**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        if env_dict is not None:
            env.update(env_dict)
        module._sanitize_pythonpath_env(env)
        if prefix is None:
            task = {
                "kind": "server",
                "function": self.child_process_fxn,
                "args": (env_dict, model, vllm_serve_args),
            }
        else:
            task = {
                "kind": "command",
                "command": ["vllm", *prefix, model, *vllm_serve_args],
            }
        print(f"Launching {type(self).__name__} in a Windows test job.", flush=True)
        self.proc = TestProcess(task, env)

    return start


def adapt_helpers(module):
    required = (
        "RemoteVLLMServer",
        "RemoteOpenAIServer",
        "RemoteLaunchRenderServer",
        "RemoteOpenAIServerCustom",
        "fork_new_process_for_each_test",
        "spawn_new_process_for_each_test",
        "requires_spawn_multiprocessing",
        "_sanitize_pythonpath_env",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Unsupported fork test helper layout; missing {missing}")
    module.requires_spawn_multiprocessing = lambda: True
    module.fork_new_process_for_each_test = spawn_test
    module.spawn_new_process_for_each_test = spawn_test
    module.RemoteVLLMServer._terminate_process_tree = _terminate_server
    for name, prefix in (
        ("RemoteOpenAIServer", ["serve"]),
        ("RemoteLaunchRenderServer", ["launch", "render"]),
        ("RemoteOpenAIServerCustom", None),
    ):
        cls = getattr(module, name)
        cls._start_server = _server_launcher(module, prefix)
        cls._terminate_process_tree = _terminate_server
    APPLIED_HELPERS[:] = [
        "tests.utils.process_isolation",
        "tests.utils.server_process_tree",
    ]


class _HelperLoader:
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        adapt_helpers(module)


class _HelperFinder:
    def find_spec(self, fullname, path=None, target=None):
        source = os.environ.get(SOURCE_ENV)
        if fullname != "tests.utils" or not source:
            return None
        spec = PathFinder.find_spec(fullname, path)
        expected = Path(source).resolve() / "tests" / "utils.py"
        if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
            raise ImportError(f"Expected test helpers from {expected}.")
        spec.loader = _HelperLoader(spec.loader)
        return spec


if sys.platform == "win32":
    sys.meta_path.insert(0, _HelperFinder())

if __name__ == "__main__":
    if sys.argv[1:] != ["--child"]:
        raise SystemExit("This module is only a Windows test subprocess entrypoint.")
    raise SystemExit(_run_child())
