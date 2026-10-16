import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from buildkite.windows import ci, run_tests


def test_public_data_dependencies_are_installed_from_index():
    from packaging.requirements import Requirement

    requirements = {}
    for raw in (ci.HERE / "requirements-tests.txt").read_text().splitlines():
        if line := raw.split("#", 1)[0].strip():
            requirement = Requirement(line)
            requirements[requirement.name] = requirement
    for name, version in (
        ("accelerate", "1.15.0"),
        ("pandas", "3.0.6"),
        ("soundfile", "0.14.0"),
    ):
        assert str(requirements[name].specifier) == f"=={version}"
        assert requirements[name].url is None
    assert {
        "peft",
        "sentence-transformers",
        "timm",
        "tensorizer",
        "cohere",
        "schemathesis",
        "imagehash",
        "multiprocess",
        "pqdm",
        "rapidfuzz",
        "msgpack",
        "orjson",
        "hf-xet",
        "pywin32",
    } <= requirements.keys()
    assert (
        not {
            "torch",
            "torchvision",
            "torchaudio",
            "transformers",
            "tokenizers",
            "numba",
            "ray",
            "datasets",
            "mteb",
            "lm-eval",
            "cohere_melody",
            "flashinfer-python",
            "pytest-forked",
        }
        & requirements.keys()
    )


@pytest.mark.parametrize("winerror", [5, 32, 33])
@pytest.mark.parametrize("failures", [1, len(run_tests.CHECKPOINT_RETRY_DELAYS)])
def test_checkpoint_retries_windows_contention_atomically(
    tmp_path, monkeypatch, capsys, winerror, failures
):
    results = run_tests.TestResults(tmp_path, tmp_path / "source")
    destination = tmp_path / "pytest-summary.json"
    previous = destination.read_bytes()
    original_replace = Path.replace
    replacements = []
    sleeps = Mock()
    error = PermissionError("Temporarily locked checkpoint")
    error.winerror = winerror

    def replace(source, target):
        replacements.append((source, target))
        assert destination.read_bytes() == previous
        assert json.loads(source.read_text())["completed"] == 1
        if len(replacements) <= failures:
            raise error
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(run_tests.time, "sleep", sleeps)
    results.update(completed=1)
    assert len(replacements) == failures + 1
    assert [call.args[0] for call in sleeps.call_args_list] == list(
        run_tests.CHECKPOINT_RETRY_DELAYS[:failures]
    )
    assert json.loads(destination.read_text())["completed"] == 1
    assert not (tmp_path / "pytest-summary.json.tmp").exists()
    assert capsys.readouterr().err.count("retrying without removing") == 1


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_checkpoint_permanent_lock_is_bounded_and_preserves_both_versions(
    tmp_path, monkeypatch, winerror
):
    results = run_tests.TestResults(tmp_path, tmp_path / "source")
    destination = tmp_path / "pytest-summary.json"
    previous = destination.read_bytes()
    error = PermissionError("Persistent checkpoint lock")
    error.winerror = winerror
    replace = Mock(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(run_tests.time, "sleep", sleep)

    with pytest.raises(PermissionError) as raised:
        results.update(completed=7)
    assert raised.value is error
    assert replace.call_count == len(run_tests.CHECKPOINT_RETRY_DELAYS) + 1
    assert sleep.call_count == len(run_tests.CHECKPOINT_RETRY_DELAYS)
    assert destination.read_bytes() == previous
    pending = tmp_path / "pytest-summary.json.tmp"
    assert json.loads(pending.read_text())["completed"] == 7
    assert "retries exhausted" in error.__notes__[-1]
    assert str(pending) in error.__notes__[-1]


@pytest.mark.parametrize("winerror", [None, 2, 6, 112])
def test_checkpoint_does_not_retry_other_io_errors(tmp_path, monkeypatch, winerror):
    results = run_tests.TestResults(tmp_path, tmp_path / "source")
    error = OSError("Unrelated I/O failure")
    if winerror is not None:
        error.winerror = winerror
    replace = Mock(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(run_tests.time, "sleep", sleep)
    with pytest.raises(OSError) as raised:
        results.update(completed=1)
    assert raised.value is error
    replace.assert_called_once()
    sleep.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="Uses a native Windows file lock")
def test_checkpoint_recovers_when_windows_reader_releases_handle(tmp_path, monkeypatch):
    results = run_tests.TestResults(tmp_path, tmp_path / "source")
    destination = tmp_path / "pytest-summary.json"
    previous = destination.read_bytes()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    # Share reads/writes but not deletion, as an ordinary Windows reader can do.
    handle = kernel.CreateFileW(str(destination), 0x80000000, 3, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.WinError(ctypes.get_last_error())
    delays = []

    def release_reader(delay):
        nonlocal handle
        delays.append(delay)
        assert destination.read_bytes() == previous
        assert (
            json.loads((tmp_path / "pytest-summary.json.tmp").read_text())["completed"]
            == 1
        )
        assert kernel.CloseHandle(handle)
        handle = None

    monkeypatch.setattr(run_tests.time, "sleep", release_reader)
    try:
        results.update(completed=1)
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
    assert delays == [run_tests.CHECKPOINT_RETRY_DELAYS[0]]
    assert json.loads(destination.read_text())["completed"] == 1
    assert not (tmp_path / "pytest-summary.json.tmp").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises native MSVC env setup")
@pytest.mark.parametrize("architecture", ["x64", "arm64"])
@pytest.mark.parametrize("failure", [None, "exit", "target", "headers", "compiler"])
def test_compiler_setup_propagates_environment_without_compiling(
    tmp_path, monkeypatch, architecture, failure
):
    tools = tmp_path / "tools with spaces & punctuation"
    tools.mkdir()
    shutil.copyfile(ci.HERE / "test-environment.cmd", tools / "test-environment.cmd")
    monkeypatch.setattr(ci, "HERE", tools)
    compiler = tools / "cl.exe"
    compiler.touch()
    vcvars = tools / "vcvarsall.bat"
    vcvars.write_text(
        "@echo off\n"
        f'echo %*>"{tools / "args.txt"}"\n'
        + ("exit /b 17\n" if failure == "exit" else "")
        + f'set "VSCMD_ARG_TGT_ARCH={"incorrect" if failure == "target" else architecture}"\n'
        + f'set "INCLUDE={"" if failure == "headers" else "test-headers"}"\n'
        + 'set "LIB=test-libraries"\n'
        + f'set "PATH={tools};%PATH%"\n'
        + "exit /b 0\n"
    )
    original = {
        **os.environ,
        "VLLM_WINDOWS_VS_PATH": str(vcvars),
        "VLLM_WINDOWS_VCVARSALL": str(vcvars),
        "CC": "wrong-compiler.exe",
        "CXX": "wrong-compiler.exe",
        "TEST_UNCHANGED": "value=with=equals",
    }
    before = dict(original)
    if failure == "compiler":
        monkeypatch.setattr(ci.shutil, "which", lambda *args, **kwargs: None)
    if failure:
        with pytest.raises(RuntimeError, match="setup failed|did not initialize"):
            ci.test_compiler_environment(architecture, original)
    else:
        prepared = ci.test_compiler_environment(architecture, original)
        assert prepared["CC"] == str(compiler)
        assert prepared["CXX"] == str(compiler)
        assert prepared["INCLUDE"] == "test-headers"
        assert prepared["LIB"] == "test-libraries"
        assert prepared["TEST_UNCHANGED"] == "value=with=equals"
        assert not any(name.startswith("VLLM_TEST_") for name in prepared)
        assert (tools / "args.txt").read_text().strip() == (
            "arm64 10.0.26100.0 -vcvars_ver=14.51.36231"
            if architecture == "arm64"
            else "x64"
        )
    assert original == before


@pytest.mark.parametrize(
    "variable,value",
    [
        ("VLLM_WINDOWS_MSVC_VERSION", "14.51 & unwanted"),
        ("VLLM_WINDOWS_SDK_VERSION", "10.0.26100.0 & unwanted"),
    ],
)
def test_compiler_rejects_invalid_version_arguments(tmp_path, variable, value):
    vcvars = tmp_path / "vcvarsall.bat"
    vcvars.touch()
    env = {"VLLM_WINDOWS_VS_PATH": str(vcvars), variable: value}
    with pytest.raises(ValueError, match=variable):
        ci.test_compiler_environment("arm64", env)


@pytest.mark.parametrize(
    "use_wheelhouse,no_index", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("architecture", ["x64", "arm64"])
@pytest.mark.parametrize("test_suite", ["full", "deepgemm"])
@pytest.mark.parametrize(
    "failed_phase",
    [
        None,
        "compiler",
        "harness",
        "wheel",
        "utilities",
        "runtime-missing",
        "runtime-change",
        "pytest",
    ],
)
def test_full_suite_setup_preserves_pool_and_failure(
    tmp_path,
    monkeypatch,
    use_wheelhouse,
    no_index,
    architecture,
    failed_phase,
    test_suite,
):
    pool = tmp_path / "pool"
    source = tmp_path / "source"
    utilities = source / "tests" / "vllm_test_utils"
    utilities.mkdir(parents=True)
    (utilities / "setup.py").touch()
    output = tmp_path / "artifacts"
    output.mkdir()
    wheel = tmp_path / "vllm.whl"
    original = {
        "PATH": "original",
        "PYTHONPATH": "unrelated-source",
        "PYTEST_ADDOPTS": "-x -k filtered",
        "VLLM_WINDOWS_NO_INDEX": "1" if no_index else "0",
    }
    if use_wheelhouse:
        original["VLLM_WINDOWS_WHEELHOUSE"] = str(tmp_path / "wheelhouse with spaces")
    calls = []
    runtime_queries = 0
    runtime = {"torch": "2.16.0.dev20261009+cu134", "numpy": "2.5.3"}
    if architecture == "arm64":
        runtime["torchvision"] = "0.29.0a0+10f68db"
    compiler_env = {
        **original,
        "PATH": "compiler-path",
        "CC": "cl.exe",
        "CXX": "cl.exe",
        "INCLUDE": "msvc-include",
        "LIB": "msvc-lib",
    }

    def compiler_environment(target, env):
        assert target == architecture
        assert env is original
        if failed_phase == "compiler":
            raise RuntimeError("Compiler setup failed")
        return compiler_env

    monkeypatch.setattr(ci, "test_compiler_environment", compiler_environment)

    def run(args, **kwargs):
        nonlocal runtime_queries
        args = [str(arg) for arg in args]
        calls.append((args, kwargs))
        if "-c" in args and kwargs.get("capture"):
            runtime_queries += 1
            if runtime_queries == 1:
                return json.dumps({} if failed_phase == "runtime-missing" else runtime)
            installed = {**runtime, "accelerate": "1.15.0"}
            if failed_phase == "runtime-change":
                installed["torch"] = "2.13.0"
            return json.dumps(installed)
        if "venv" in args:
            (Path(args[-1]) / "Lib" / "site-packages").mkdir(parents=True)
        phase = (
            "pytest"
            if str(ci.HERE / "run_tests.py") in args
            else (
                "harness"
                if "-r" in args
                else "utilities"
                if str(utilities) in args
                else "wheel"
                if "--no-deps" in args
                else "venv"
            )
        )
        if phase == failed_phase:
            raise subprocess.CalledProcessError(17, args)

    monkeypatch.setattr(ci, "run", run)
    args = (
        pool / "Scripts" / "python.exe",
        pool,
        source,
        wheel,
        tmp_path,
        output,
        original,
        architecture,
        test_suite,
    )
    if failed_phase == "compiler":
        with pytest.raises(RuntimeError, match="Compiler setup failed"):
            ci.run_full_tests(*args)
        assert not calls
        assert not (tmp_path / "test-venv").exists()
        assert json.loads((output / "pytest-summary.json").read_text())["suite"] == (
            test_suite
        )
        return
    if failed_phase in ("runtime-change", "runtime-missing"):
        message = (
            "changed protected runtime packages"
            if failed_phase == "runtime-change"
            else "cannot see the provisioned Torch distribution"
        )
        with pytest.raises(RuntimeError, match=message):
            ci.run_full_tests(*args)
        assert not any(str(ci.HERE / "run_tests.py") in command for command, _ in calls)
        if failed_phase == "runtime-change":
            report = json.loads((output / "test-dependencies.json").read_text())
            assert report["protected_runtime"] == runtime
            assert report["installed"]["torch"] == "2.13.0"
        else:
            assert not any("-r" in command for command, _ in calls)
        return
    if failed_phase:
        with pytest.raises(subprocess.CalledProcessError) as error:
            ci.run_full_tests(*args)
        assert error.value.returncode == 17
    else:
        ci.run_full_tests(*args)
    assert original["PYTHONPATH"] == "unrelated-source"
    assert original["PYTEST_ADDOPTS"] == "-x -k filtered"
    inherited = (
        tmp_path / "test-venv" / "Lib" / "site-packages" / "pool-dependencies.pth"
    )
    assert inherited.read_text().strip() == str(pool / "Lib" / "site-packages")
    for command, kwargs in calls[1:]:
        assert command[0] == str(tmp_path / "test-venv" / "Scripts" / "python.exe")
        assert "PYTHONPATH" not in kwargs["env"]
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        assert kwargs["env"]["VIRTUAL_ENV"] == str(tmp_path / "test-venv")
        assert kwargs["env"]["CC"] == "cl.exe"
        assert kwargs["env"]["INCLUDE"] == "msvc-include"
        assert kwargs["env"]["PATH"].endswith(os.pathsep + "compiler-path")
    if failed_phase == "wheel":
        assert len(calls) == 2
        return
    if failed_phase == "utilities":
        assert len(calls) == 3
        return
    harness = calls[4][0]
    assert "--only-binary=:all:" in harness
    assert ("--find-links" in harness) == use_wheelhouse
    if use_wheelhouse:
        assert (
            harness[harness.index("--find-links") + 1]
            == original["VLLM_WINDOWS_WHEELHOUSE"]
        )
    assert ("--no-index" in harness) == no_index
    assert harness[harness.index("-c") + 1] == str(
        output / "test-runtime-constraints.txt"
    )
    assert (output / "test-runtime-constraints.txt").read_text().splitlines() == [
        f"{name}=={version}" for name, version in sorted(runtime.items())
    ]
    if failed_phase == "harness":
        assert len(calls) == 5
    if not failed_phase or failed_phase == "pytest":
        assert len(calls) == 7
        assert runtime_queries == 2
        assert "--no-deps" in calls[1][0]
        assert "--only-binary=:all:" in calls[1][0]
        assert "--no-compile" in calls[1][0]
        assert "--ignore-installed" in calls[1][0]
        assert str(wheel) in calls[1][0]
        torchvision = (
            ci.HERE
            / "wheelhouse"
            / "torchvision-0.29.0a0+10f68db-cp313-cp313-win_arm64.whl"
        )
        assert (str(torchvision) in calls[1][0]) == (architecture == "arm64")
        assert not any("flashinfer" in arg for arg in calls[1][0])
        assert str(utilities) in calls[2][0]
        assert "--no-deps" in calls[2][0]
        assert "--no-build-isolation" in calls[2][0]
        report = json.loads((output / "test-dependencies.json").read_text())
        assert report["protected_runtime"] == runtime
        assert report["installed"] == {**runtime, "accelerate": "1.15.0"}
        assert calls[-1][0][1:3] == ["-I", str(ci.HERE / "run_tests.py")]
        assert calls[-1][0][-2:] == ["--test-suite", test_suite]
        assert calls[-1][1]["cwd"] == source
        assert (inherited.parent / "_vllm_ci_wheel_imports.py").read_text() == (
            ci.HERE / "wheel_imports.py"
        ).read_text()
        assert (inherited.parent / "vllm-ci-wheel.pth").read_text().strip() == (
            "import _vllm_ci_wheel_imports; import _vllm_ci_windows_helpers"
        )
        assert (inherited.parent / "_vllm_ci_windows_helpers.py").read_text() == (
            ci.HERE / "windows_helpers.py"
        ).read_text()


def test_missing_bundled_torchvision_fails_before_environment_setup(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ci, "HERE", tmp_path / "ci")
    compiler = Mock()
    runner = Mock()
    monkeypatch.setattr(ci, "test_compiler_environment", compiler)
    monkeypatch.setattr(ci, "run", runner)
    with pytest.raises(
        FileNotFoundError, match="Bundled ARM64 TorchVision wheel is missing"
    ):
        ci.run_full_tests(
            tmp_path / "python.exe",
            tmp_path / "pool",
            tmp_path / "source",
            tmp_path / "vllm.whl",
            tmp_path,
            tmp_path,
            {},
            "arm64",
        )
    compiler.assert_not_called()
    runner.assert_not_called()


def test_missing_source_test_utilities_fail_before_setup(tmp_path, monkeypatch):
    compiler = Mock()
    runner = Mock()
    monkeypatch.setattr(ci, "test_compiler_environment", compiler)
    monkeypatch.setattr(ci, "run", runner)
    with pytest.raises(FileNotFoundError, match="no test utility package"):
        ci.run_full_tests(
            tmp_path / "python.exe",
            tmp_path / "pool",
            tmp_path / "source",
            tmp_path / "vllm.whl",
            tmp_path,
            tmp_path,
            {},
            "arm64",
        )
    compiler.assert_not_called()
    runner.assert_not_called()


def test_full_runner_rejects_cached_or_source_package(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "tests").mkdir(parents=True)
    monkeypatch.setattr(
        run_tests,
        "sys",
        SimpleNamespace(
            prefix=str(tmp_path / "venv"), path=[], stdout=sys.stdout, stderr=sys.stderr
        ),
    )
    importer = Mock(
        return_value=SimpleNamespace(__file__=str(source / "vllm" / "__init__.py"))
    )
    monkeypatch.setattr(run_tests, "importlib", SimpleNamespace(import_module=importer))
    with pytest.raises(RuntimeError, match="job's installed wheel"):
        run_tests.run_tests(source, tmp_path)
    importer.assert_called_once_with("vllm")
    summary = json.loads((tmp_path / "pytest-summary.json").read_text())
    assert summary["status"] == "error"
    assert summary["exit_code"] is None
    failure = json.loads((tmp_path / "pytest-failures.jsonl").read_text())
    assert failure["phase"] == "runner"
    assert "RuntimeError" in failure["traceback"]


def prepare_full_runner(tmp_path):
    venv = tmp_path / "test-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True
    )
    if sys.platform == "win32":
        python = venv / "Scripts" / "python.exe"
        site = venv / "Lib" / "site-packages"
    else:
        python = venv / "bin" / "python"
        site = (
            venv
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
    # Use the existing test dependencies, never download or install anything.
    inherited = [
        path
        for path in sys.path
        if path and ("site-packages" in path or "dist-packages" in path)
    ]
    (site / "test-harness.pth").write_text("\n".join(inherited) + "\n")
    shutil.copyfile(ci.HERE / "wheel_imports.py", site / "_vllm_ci_wheel_imports.py")
    shutil.copyfile(
        ci.HERE / "windows_helpers.py", site / "_vllm_ci_windows_helpers.py"
    )
    (site / "vllm-ci-wheel.pth").write_text(
        "import _vllm_ci_wheel_imports; import _vllm_ci_windows_helpers\n"
    )
    installed = site / "vllm"
    installed.mkdir()
    (installed / "__init__.py").write_text("ORIGIN = 'built-wheel'\n")
    source = tmp_path / "source"
    tests = source / "tests"
    tests.mkdir(parents=True)
    (source / "vllm").mkdir()
    (source / "vllm" / "__init__.py").write_text(
        "raise RuntimeError('Imported source instead of wheel')\n"
    )
    (source / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-x -k nonexistent"\n'
        'pythonpath = ["."]\nmarkers = ["optional"]\n'
    )
    (tests / "__init__.py").touch()
    (tests / "helper.py").write_text("VALUE = 17\n")
    (tests / "conftest.py").write_text(
        "import pytest\n"
        "def pytest_addoption(parser):\n"
        "    parser.addoption('--optional', action='store_true', default=False)\n"
        "def pytest_collection_modifyitems(config, items):\n"
        "    if not config.getoption('--optional'):\n"
        "        for item in items:\n"
        "            if 'optional' in item.keywords:\n"
        "                item.add_marker(pytest.mark.skip(reason='optional disabled'))\n"
    )
    output = tmp_path / "artifacts"
    output.mkdir()
    env = dict(os.environ)
    env["PYTEST_ADDOPTS"] = "-x -k nonexistent"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return python, source, tests, output, env


def test_dependency_snapshot_prefers_local_versions_over_shared_pool(tmp_path):
    python, _source, _tests, _output, env = prepare_full_runner(tmp_path)
    site = next(python.parent.parent.rglob("_vllm_ci_wheel_imports.py")).parent
    shared = tmp_path / "shared-packages"
    for directory, version in ((site, "2.0"), (shared, "1.0")):
        metadata = directory / f"ci_runtime_probe-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: ci_runtime_probe\nVersion: {version}\n"
        )
    (site / "shared-probe.pth").write_text(str(shared) + "\n")
    packages = ci.installed_test_packages(python, env)
    assert packages["ci-runtime-probe"] == "2.0"
    assert "ci_runtime_probe" not in packages


@pytest.mark.parametrize("test_failure", [False, True])
def test_checkpoint_contention_does_not_interrupt_pytest_or_hide_failures(
    tmp_path, test_failure
):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    (tests / "test_checkpoint.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        f"checkpoint = Path({str(output / 'pytest-summary.json')!r})\n"
        "def test_a_contended_checkpoint():\n"
        "    previous = checkpoint.read_bytes()\n"
        "    original_replace = Path.replace\n"
        "    def transient_lock(source, target):\n"
        "        if target != checkpoint:\n"
        "            return original_replace(source, target)\n"
        "        assert checkpoint.read_bytes() == previous\n"
        "        Path.replace = original_replace\n"
        "        error = PermissionError('Temporary reader lock')\n"
        "        error.winerror = 5\n"
        "        raise error\n"
        "    Path.replace = transient_lock\n"
        f"    assert {not test_failure!r}, 'original test failure'\n"
        "def test_b_runs_after_contention():\n"
        "    summary = json.loads(checkpoint.read_text())\n"
        "    assert summary['completed'] == 1\n"
        f"    assert summary['failed'] == {int(test_failure)}\n"
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
    )
    assert result.returncode == int(test_failure), result.stdout + result.stderr
    summary = json.loads((output / "pytest-summary.json").read_text())
    assert summary["completed"] == summary["collected"] == 2
    assert summary["passed"] == 2 - int(test_failure)
    assert summary["failed"] == int(test_failure)
    assert summary["status"] == ("failed" if test_failure else "passed")
    assert summary["exit_code"] == int(test_failure)
    assert "retrying without removing" in result.stdout + result.stderr
    assert "INTERNALERROR>" not in result.stdout + result.stderr
    if test_failure:
        event = json.loads((output / "pytest-failures.jsonl").read_text())
        assert "original test failure" in event["traceback"]


@pytest.mark.parametrize("collection_failure", [False, True])
@pytest.mark.parametrize("test_failure", [False, True])
def test_full_runner_really_runs_optional_tests_without_filters(
    tmp_path, collection_failure, test_failure
):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    (tests / "test_all.py").write_text(
        "import os, subprocess, sys, pytest, vllm\n"
        "from tests.helper import VALUE\n"
        "def test_installed_package():\n"
        "    assert vllm.ORIGIN == 'built-wheel'\n"
        "    assert VALUE == 17\n"
        "@pytest.mark.optional\n"
        "def test_optional():\n"
        "    with pytest.raises(RuntimeError, match='CUDA error: out of memory'):\n"
        "        raise RuntimeError('CUDA error: out of memory')\n"
        "def test_child_imports_installed_package():\n"
        "    subprocess.run([sys.executable, '-c', "
        '"import os, sys; sys.path.insert(0, os.getcwd()); '
        "import vllm; assert vllm.ORIGIN == 'built-wheel'\"], check=True)\n"
    )
    if collection_failure:
        (tests / "test_missing_dependency.py").write_text(
            "import missing_windows_backend_for_test\n"
        )
    if test_failure:
        (tests / "test_failure.py").write_text(
            "import json, pathlib\n"
            "def test_a_failure():\n"
            "    print('captured diagnostic output')\n"
            "    assert False, 'deliberate live failure'\n"
            "def test_b_runs_after_failure():\n"
            f"    output = pathlib.Path({str(output)!r})\n"
            "    summary = json.loads((output / 'pytest-summary.json').read_text())\n"
            "    assert summary['status'] == 'running'\n"
            "    assert summary['failed'] == 1\n"
            "    assert summary['active_test'].endswith('test_b_runs_after_failure')\n"
            "    failures = (output / 'pytest-failures.jsonl').read_text()\n"
            "    assert 'deliberate live failure' in failures\n"
            "    assert 'captured diagnostic output' in failures\n"
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
    )
    assert result.returncode == (1 if collection_failure or test_failure else 0), (
        result.stdout + result.stderr
    )
    summary = json.loads((output / "pytest-summary.json").read_text())
    assert summary["suite"] == "full"
    assert summary["selected_paths"] == ["tests"]
    assert summary["cuda_launch_blocking"] == "0"
    assert "--tb=short" in summary["arguments"]
    assert summary["collected"] == 3 + 2 * int(test_failure)
    assert summary["passed"] == 3 + int(test_failure)
    assert summary["failed"] == int(test_failure)
    assert summary["deselected"] == 0
    assert summary["skipped"] == 0
    assert summary["collection_errors"] == int(collection_failure)
    assert summary["exit_code"] == result.returncode
    assert summary["status"] == (
        "failed" if test_failure or collection_failure else "passed"
    )
    assert summary["completed"] == summary["collected"]
    assert summary["active_test"] is None
    assert (output / "pytest.xml").is_file()
    journal = output / "pytest-failures.jsonl"
    if not test_failure and not collection_failure:
        assert not journal.exists()
    else:
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        assert len(events) == int(test_failure) + int(collection_failure)
        if test_failure:
            assert "CALL FAILED: tests/test_failure.py::test_a_failure" in result.stdout
            assert "deliberate live failure" in result.stdout
            assert any(
                name == "Captured stdout call" for name, _ in events[-1]["sections"]
            )
        if collection_failure:
            assert events[0]["phase"] == "collection"
            assert "missing_windows_backend_for_test" in events[0]["traceback"]
            assert "COLLECTION FAILED:" in result.stdout


def test_child_import_hook_rejects_missing_wheel_instead_of_source(tmp_path):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    site = next((python.parent.parent).rglob("_vllm_ci_wheel_imports.py")).parent
    (site / "vllm").rename(site / "removed-vllm")
    result = subprocess.run(
        [str(python), "-c", "import vllm"],
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "refusing source fallback" in result.stderr
    assert "Imported source instead of wheel" not in result.stderr


@pytest.mark.parametrize(
    "module_state", ["present", "missing", "collection-error", "cuda-failure"]
)
def test_deepgemm_runner_only_collects_selected_module(tmp_path, module_state):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    env["CUDA_LAUNCH_BLOCKING"] = "0"
    for directory in ("distributed", "basic_correctness", "entrypoints"):
        excluded = tests / directory
        excluded.mkdir()
        (excluded / "conftest.py").write_text(
            "raise RuntimeError('Must not import excluded suite conftest')\n"
        )
        (excluded / "test_excluded.py").write_text(
            "raise RuntimeError('Must not collect memory-budget or distributed tests')\n"
        )
    target = source / run_tests.TEST_TARGETS["deepgemm"]
    if module_state != "missing":
        target.parent.mkdir(parents=True)
        target.write_text(
            "import os, pytest, vllm\n"
            "assert os.environ['CUDA_LAUNCH_BLOCKING'] == '1'\n"
            "assert vllm.ORIGIN == 'built-wheel'\n"
            + (
                "raise ImportError('DeepGEMM dependency is missing')\n"
                if module_state == "collection-error"
                else "@pytest.mark.parametrize('case', [0, 1])\n"
                "@pytest.mark.optional\n"
                "def test_deepgemm(case):\n"
                + (
                    "    raise RuntimeError('CUDA error: unspecified launch failure')\n"
                    if module_state == "cuda-failure"
                    else "    assert case in (0, 1)\n"
                )
            )
        )
    original = target.read_bytes() if target.is_file() else None
    result = subprocess.run(
        [
            str(python),
            "-I",
            str(Path(run_tests.__file__)),
            "--source-dir",
            str(source),
            "--output",
            str(output),
            "--test-suite",
            "deepgemm",
        ],
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    summary = json.loads((output / "pytest-summary.json").read_text())
    assert summary["suite"] == "deepgemm"
    assert summary["selected_paths"] == [str(run_tests.TEST_TARGETS["deepgemm"])]
    assert "Must not" not in result.stdout + result.stderr
    assert summary["skipped"] == summary["deselected"] == 0
    if module_state == "missing":
        assert result.returncode != 0
        assert summary["status"] == "error"
        assert "no DeepGEMM test module" in result.stderr
        assert not target.exists()
        return
    assert target.read_bytes() == original
    assert summary["cuda_launch_blocking"] == "1"
    assert summary["arguments"][0] == str(target)
    assert "--tb=long" in summary["arguments"]
    assert "-vv" in summary["arguments"]
    assert (output / "pytest.xml").is_file()
    if module_state == "cuda-failure":
        assert result.returncode == 1, result.stdout + result.stderr
        assert summary["status"] == "aborted"
        assert summary["collected"] == 2
        assert summary["completed"] == summary["failed"] == 1
        assert summary["passed"] == 0
        assert "test_deepgemm[0]" in summary["stop_reason"]
        assert summary["collection_errors"] == 0
    elif module_state == "collection-error":
        assert result.returncode == 1, result.stdout + result.stderr
        assert summary["status"] == "failed"
        assert summary["collection_errors"] == 1
        assert summary["collected"] == 0
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert summary["status"] == "passed"
        assert summary["collected"] == summary["completed"] == summary["passed"] == 2
        assert summary["collection_errors"] == 0


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: out of memory",
        "CUDA error: an illegal memory access was encountered",
        "CUDA error: unspecified launch failure",
        "CUDA_ERROR_LAUNCH_FAILED",
        "CUDA error: device-side assert triggered",
        "CUDA_ERROR_OUT_OF_MEMORY",
        "Sticky error detected",
    ],
)
@pytest.mark.parametrize("phase", ["call", "setup", "teardown"])
def test_cuda_failure_stops_after_teardown_and_records_incomplete_suite(
    tmp_path, message, phase
):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    (tests / "test_cuda_failure.py").write_text(
        "import pathlib, pytest\n"
        f"output = pathlib.Path({str(output)!r})\n"
        "@pytest.fixture(autouse=True)\n"
        "def cleanup(request):\n"
        "    request.addfinalizer(lambda: (output / 'teardown.txt').write_text('done'))\n"
        + (f"    raise RuntimeError({message!r})\n" if phase == "setup" else "")
        + "    yield\n"
        + (f"    raise RuntimeError({message!r})\n" if phase == "teardown" else "")
        + "def test_a_cuda_failure():\n"
        + (
            f"    raise RuntimeError({message!r})\n"
            if phase == "call"
            else "    pass\n"
        )
        + "def test_b_must_not_run():\n"
        "    (output / 'unexpected.txt').write_text('cascaded')\n"
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
    )
    assert result.returncode == 1, result.stdout + result.stderr
    summary = json.loads((output / "pytest-summary.json").read_text())
    assert summary["status"] == "aborted"
    assert summary["collected"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["exit_code"] == 1
    assert "test_a_cuda_failure" in summary["stop_reason"]
    assert summary["active_test"] is None
    assert (output / "teardown.txt").read_text() == "done"
    assert not (output / "unexpected.txt").exists()
    events = [
        json.loads(line)
        for line in (output / "pytest-failures.jsonl").read_text().splitlines()
    ]
    assert events[0]["phase"] == phase
    assert message in events[0]["traceback"]
    assert (output / "pytest.xml").is_file()


@pytest.mark.parametrize("termination", ["hard-exit", "interrupt"])
def test_interrupted_suite_preserves_live_failures_and_progress(tmp_path, termination):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    (tests / "test_diagnostics.py").write_text(
        "import json, os, pathlib, pytest\n"
        f"output = pathlib.Path({str(output)!r})\n"
        "def test_a_failure():\n"
        "    print('keep this captured output')\n"
        "    assert False, 'visible before termination'\n"
        "def test_b_terminate():\n"
        "    summary = json.loads((output / 'pytest-summary.json').read_text())\n"
        "    assert summary['failed'] == 1\n"
        "    assert summary['active_test'].endswith('test_b_terminate')\n"
        "    assert 'visible before termination' in (output / 'pytest-failures.jsonl').read_text()\n"
        + (
            "    os._exit(23)\n"
            if termination == "hard-exit"
            else "    pytest.exit('interrupted test run', returncode=2)\n"
        )
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
    )
    assert result.returncode == (23 if termination == "hard-exit" else 2), (
        result.stdout + result.stderr
    )
    summary = json.loads((output / "pytest-summary.json").read_text())
    assert summary["failed"] == 1
    assert summary["collected"] == 2
    assert summary["completed"] == 1
    assert summary["active_test"].endswith("test_b_terminate")
    assert summary["status"] == (
        "running" if termination == "hard-exit" else "interrupted"
    )
    assert summary["exit_code"] == (None if termination == "hard-exit" else 2)
    failure = json.loads((output / "pytest-failures.jsonl").read_text().splitlines()[0])
    assert failure["nodeid"].endswith("test_a_failure")
    assert failure["phase"] == "call"
    assert "visible before termination" in result.stdout
    assert "keep this captured output" in result.stdout


@pytest.mark.parametrize("phase", ["setup", "teardown"])
def test_live_diagnostics_include_fixture_errors(tmp_path, phase):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    body = (
        "    raise RuntimeError('broken setup fixture')\n    yield\n"
        if phase == "setup"
        else "    yield\n    raise RuntimeError('broken teardown fixture')\n"
    )
    (tests / "test_fixture_error.py").write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def broken():\n" + body + "def test_fixture(broken):\n    assert True\n"
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
    )
    assert result.returncode == 1, result.stdout + result.stderr
    event = json.loads((output / "pytest-failures.jsonl").read_text())
    assert event["phase"] == phase
    assert f"broken {phase} fixture" in event["traceback"]
    assert f"{phase.upper()} FAILED:" in result.stdout


def test_live_diagnostics_preserve_pytest_internal_error(tmp_path):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    with (tests / "conftest.py").open("a") as stream:
        stream.write(
            "def pytest_collection_finish(session):\n"
            "    raise RuntimeError('broken collection hook')\n"
        )
    (tests / "test_collected.py").write_text("def test_one():\n    assert True\n")
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
    )
    assert result.returncode == 3, result.stdout + result.stderr
    summary = json.loads((output / "pytest-summary.json").read_text())
    assert summary["exit_code"] == 3
    assert summary["status"] == "error"
    event = json.loads((output / "pytest-failures.jsonl").read_text())
    assert event["phase"] == "internal"
    assert "broken collection hook" in event["traceback"]
    assert "INTERNAL FAILED:" in result.stdout
