import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from buildkite.windows import ci, smoke

ROOT = Path(__file__).resolve().parents[2]
SHA = "a" * 40
SOURCE = {"repository": ci.REPOSITORY, "branch": ci.DEFAULT_BRANCH, "commit": SHA}


def make_wheel(directory, architecture="x64", *, tag=None, extension=True):
    platform = ci.PLATFORMS[architecture]
    wheel = directory / f"vllm-0.29.0-cp313-cp313-{platform}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "vllm-0.29.0.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nTag: {tag or f'cp313-cp313-{platform}'}\n",
        )
        if extension:
            archive.writestr("vllm/_C.pyd", b"test extension")
    return wheel


def provisioned_environment(tmp_path):
    venv = tmp_path / "private venv"
    cuda = tmp_path / "CUDA"
    vcvars = tmp_path / "Visual Studio" / "vcvarsall.bat"
    for file in (
        venv / "Scripts" / "python.exe",
        cuda / "bin" / "nvcc.exe",
        cuda / "lib" / "x64" / "cudart.lib",
        cuda / "lib" / "arm64" / "cudart.lib",
        vcvars,
    ):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    return {
        "VLLM_WINDOWS_VENV": str(venv),
        "VLLM_WINDOWS_VCVARSALL": str(vcvars),
        "CUDA_PATH": str(cuda),
        "TORCH_CUDA_ARCH_LIST": "12.0+PTX;10.3a",
        "CMAKE_CUDA_ARCHITECTURES": "120-real;103-real",
        "VLLM_WINDOWS_WORK_ROOT": str(tmp_path / "work"),
        "COMSPEC": "cmd.exe",
        "PATH": "original-path",
    }


@pytest.mark.parametrize(
    "filename,architectures,resolver_architecture",
    [
        ("windows.yml", ("x64", "arm64"), "x64"),
        ("windows-arm64.yml", ("arm64",), "arm64"),
    ],
)
def test_pipeline_routes_selected_native_architectures(
    filename, architectures, resolver_architecture
):
    pipeline = yaml.safe_load(
        (ROOT / ".buildkite" / "pipelines" / filename).read_text()
    )
    source, *builds = pipeline["steps"]
    assert len(builds) == len(architectures)
    assert source["key"] == "windows-source"
    assert source["command"] == r"python buildkite\windows\ci.py resolve"
    variable = f"WINDOWS_{resolver_architecture.upper()}_QUEUE"
    assert source["agents"] == {
        "queue": f"${{{variable}:-windows-{resolver_architecture}}}"
    }
    for architecture, step in zip(architectures, builds):
        variable = f"WINDOWS_{architecture.upper()}_QUEUE"
        assert step["agents"] == {"queue": f"${{{variable}:-windows-{architecture}}}"}
        assert step["key"] == f"windows-cuda-{architecture}"
        assert step["depends_on"] == source["key"]
        assert step["command"] == (
            rf"python buildkite\windows\ci.py build --architecture {architecture}"
        )
        assert step["artifact_paths"] == [f"artifacts/windows-{architecture}/**/*"]
        assert step["timeout_in_minutes"] == {"x64": 240, "arm64": 360}[architecture]
        assert step["retry"] == {"automatic": [{"exit_status": -1, "limit": 1}]}
        assert "plugins" not in step
        assert not step.get("soft_fail")


def test_resolve_pins_branch_once(monkeypatch):
    monkeypatch.delenv("VLLM_WINDOWS_COMMIT", raising=False)
    monkeypatch.delenv("VLLM_WINDOWS_BRANCH", raising=False)
    monkeypatch.setattr(
        ci.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=100))
    )
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        if "ls-remote" in args:
            return f"{SHA}\trefs/heads/{ci.DEFAULT_BRANCH}"

    monkeypatch.setattr(ci, "run", run)
    ci.resolve_source()
    assert json.loads(commands[-1][-1]) == SOURCE
    assert commands[-1][:4] == ["buildkite-agent", "meta-data", "set", ci.SOURCE_KEY]
    assert commands[1][-1] == f"refs/heads/{ci.DEFAULT_BRANCH}"


def test_resolve_reuses_metadata_on_retry(monkeypatch):
    monkeypatch.setattr(
        ci.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0))
    )
    runner = Mock(return_value=json.dumps(SOURCE))
    monkeypatch.setattr(ci, "run", runner)
    ci.resolve_source()
    runner.assert_called_once_with(
        ["buildkite-agent", "meta-data", "get", ci.SOURCE_KEY], capture=True
    )


def test_resolve_explicit_commit_does_not_resolve_branch(monkeypatch):
    monkeypatch.setenv("VLLM_WINDOWS_COMMIT", SHA)
    monkeypatch.setenv("VLLM_WINDOWS_BRANCH", ci.DEFAULT_BRANCH)
    monkeypatch.setattr(
        ci.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=100))
    )
    runner = Mock()
    monkeypatch.setattr(ci, "run", runner)
    ci.resolve_source()
    assert runner.call_count == 1
    assert json.loads(runner.call_args.args[0][-1]) == SOURCE


def test_resolve_does_not_treat_agent_error_as_missing_metadata(monkeypatch):
    monkeypatch.setattr(
        ci.subprocess,
        "run",
        Mock(return_value=SimpleNamespace(returncode=1, args=["buildkite-agent"])),
    )
    runner = Mock()
    monkeypatch.setattr(ci, "run", runner)
    with pytest.raises(subprocess.CalledProcessError):
        ci.resolve_source()
    runner.assert_not_called()


@pytest.mark.parametrize(
    "source",
    [
        None,
        {},
        {**SOURCE, "commit": "HEAD"},
        {**SOURCE, "commit": SHA + "\n"},
        {**SOURCE, "repository": "https://example.com"},
    ],
)
def test_invalid_source_is_rejected(source):
    with pytest.raises(ValueError, match="Invalid Windows source"):
        ci.validate_source(source)


@pytest.mark.parametrize("architecture", ["x64", "arm64"])
def test_build_environment_requires_matching_cuda_libraries(tmp_path, architecture):
    env = provisioned_environment(tmp_path)
    (Path(env["CUDA_PATH"]) / "lib" / architecture / "cudart.lib").unlink()
    with pytest.raises(ValueError, match="Required .* CUDA file"):
        ci.build_environment(architecture, env)


def test_build_environment_does_not_reuse_precompiled_wheels(tmp_path):
    env = provisioned_environment(tmp_path)
    env.update(
        VLLM_USE_PRECOMPILED="1",
        VLLM_USE_PRECOMPILED_RUST="1",
        VLLM_PRECOMPILED_WHEEL_LOCATION="cached.whl",
        VLLM_BUILD_BASE="old-build",
    )
    python, venv, prepared = ci.build_environment("x64", env)
    assert python == venv / "Scripts" / "python.exe"
    assert prepared["VLLM_TARGET_DEVICE"] == "cuda"
    assert prepared["VLLM_USE_PRECOMPILED"] == "0"
    assert prepared["VLLM_USE_PRECOMPILED_RUST"] == "0"
    assert prepared["VLLM_REQUIRE_RUST_FRONTEND"] == "1"
    assert "VLLM_PRECOMPILED_WHEEL_LOCATION" not in prepared
    assert "VLLM_BUILD_BASE" not in prepared
    assert env["VLLM_USE_PRECOMPILED"] == "1"


@pytest.mark.parametrize(
    "variable,value",
    [
        ("TORCH_CUDA_ARCH_LIST", ""),
        ("CMAKE_CUDA_ARCHITECTURES", ""),
        ("VLLM_WINDOWS_VENV", ""),
        ("MAX_JOBS", "0"),
        ("MAX_JOBS", "257"),
    ],
)
def test_build_environment_rejects_missing_or_invalid_settings(
    tmp_path, variable, value
):
    env = provisioned_environment(tmp_path)
    env[variable] = value
    with pytest.raises(ValueError, match=variable):
        ci.build_environment("arm64", env)


def test_arm64_reuses_fork_helper_without_skipping_build(tmp_path):
    env = provisioned_environment(tmp_path)
    python, venv, env = ci.build_environment("arm64", env)
    helper = tmp_path / "tools" / "build-win-arm64.ps1"
    helper.parent.mkdir()
    helper.touch()
    env["VLLM_WINDOWS_MSVC_VERSION"] = "14.51.36231"
    command = ci.build_command("arm64", tmp_path, python, venv, tmp_path / "out", env)
    assert command[:5] == ["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-File"]
    assert command[5] == str(helper)
    assert command[command.index("-VenvDir") + 1] == str(venv)
    assert command[command.index("-MsvcToolsetVersion") + 1] == "14.51.36231"
    assert command[command.index("-CMakeCudaArchitectures") + 1] == "120-real;103-real"
    assert "-SkipBuild" not in command
    assert "-SkipRustFrontend" not in command


@pytest.mark.parametrize("architecture", ["x64", "arm64"])
def test_wheel_architecture_and_checksum(tmp_path, architecture):
    wheel = make_wheel(tmp_path, architecture)
    assert ci.validate_wheel(tmp_path, architecture) == wheel
    assert (tmp_path / f"{wheel.name}.sha256").read_text() == (
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}\n"
    )


@pytest.mark.parametrize(
    "problem", ["empty", "wrong-arch", "stale", "wrong-tag", "no-extension"]
)
def test_invalid_wheel_outputs_fail(tmp_path, problem):
    if problem == "wrong-arch":
        make_wheel(tmp_path, "arm64")
    elif problem == "stale":
        wheel = make_wheel(tmp_path)
        shutil.copyfile(wheel, tmp_path / "old-win_amd64.whl")
    elif problem == "wrong-tag":
        make_wheel(tmp_path, tag="cp313-cp313-win_arm64")
    elif problem == "no-extension":
        make_wheel(tmp_path, extension=False)
    with pytest.raises(ValueError):
        ci.validate_wheel(tmp_path, "x64")


@pytest.mark.parametrize("architecture", ["x64", "arm64"])
@pytest.mark.parametrize(
    "failure", [None, "compile", "smoke", "wrong-python", "no-gpu", "wrong-source"]
)
def test_build_lifecycle(tmp_path, monkeypatch, architecture, failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ci, "sys", SimpleNamespace(platform="win32"))
    env = provisioned_environment(tmp_path)
    prepared = ci.build_environment(architecture, env)
    monkeypatch.setattr(ci, "build_environment", lambda *args: prepared)
    monkeypatch.setattr(ci, "build_command", lambda *args: ["compile-wheel"])
    output = tmp_path / "artifacts" / f"windows-{architecture}"
    calls = []

    def run(args, **kwargs):
        args = [str(arg) for arg in args]
        calls.append((args, kwargs))
        if args[:3] == ["buildkite-agent", "meta-data", "get"]:
            return json.dumps(SOURCE)
        if "-c" in args:
            return json.dumps(
                {
                    "platform": "wrong"
                    if failure == "wrong-python"
                    else ci.PLATFORMS[architecture].replace("_", "-"),
                    "torch": "private-torch",
                    "cuda": "13.4",
                    "gpu_available": failure != "no-gpu",
                }
            )
        if "rev-parse" in args:
            return "b" * 40 if failure == "wrong-source" else SHA
        if args == ["compile-wheel"]:
            if failure == "compile":
                raise subprocess.CalledProcessError(23, args)
            make_wheel(output, architecture)
        if str(ci.HERE / "smoke.py") in args and failure == "smoke":
            raise subprocess.CalledProcessError(24, args)

    monkeypatch.setattr(ci, "run", run)
    if failure:
        with pytest.raises((subprocess.CalledProcessError, ValueError)):
            ci.build(architecture)
    else:
        ci.build(architecture)
        install = next(args for args, _ in calls if "install" in args)
        assert "--target" in install and "--no-deps" in install
        smoke_call, options = calls[-1]
        assert "-I" in smoke_call
        assert str(ci.HERE / "smoke.py") in smoke_call
        assert options["cwd"] != tmp_path
        assert not Path(options["cwd"]).exists()
        fetch = next(args for args, _ in calls if "fetch" in args)
        assert fetch[-1] == SHA
        assert "--tags" in fetch and "--depth=1" not in fetch
    if failure in ("compile", "wrong-python", "no-gpu", "wrong-source"):
        assert not any(str(ci.HERE / "smoke.py") in args for args, _ in calls)


def test_build_rejects_dirty_output_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ci, "sys", SimpleNamespace(platform="win32"))
    prepared = ci.build_environment("x64", provisioned_environment(tmp_path))
    monkeypatch.setattr(ci, "build_environment", lambda *args: prepared)
    runner = Mock(return_value=json.dumps(SOURCE))
    monkeypatch.setattr(ci, "run", runner)
    output = tmp_path / "artifacts" / "windows-x64"
    output.mkdir(parents=True)
    wheel = make_wheel(output)
    with pytest.raises(FileExistsError):
        ci.build("x64")
    assert wheel.is_file()
    assert runner.call_count == 1


def test_smoke_rejects_wrong_native_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(smoke.sysconfig, "get_platform", lambda: "win-amd64")
    with pytest.raises(RuntimeError, match="native win-arm64"):
        smoke.smoke(tmp_path, "arm64")


@pytest.mark.parametrize("cached_package", [True, False])
def test_smoke_does_not_accept_cached_package_or_missing_gpu(
    tmp_path, monkeypatch, cached_package
):
    monkeypatch.setattr(smoke, "sys", SimpleNamespace(platform="win32", path=[]))
    monkeypatch.setattr(smoke.sysconfig, "get_platform", lambda: "win-amd64")
    package = tmp_path / "installed"
    origin = tmp_path / "old" if cached_package else package
    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(__file__=str(origin / "__init__.py"))
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    with pytest.raises(
        RuntimeError, match="outside" if cached_package else "CUDA is not available"
    ):
        smoke.smoke(package, "x64")


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises native cmd.exe")
@pytest.mark.parametrize("setup_exit,build_exit", [(0, 0), (17, 0), (0, 23)])
def test_x64_cmd_propagates_failures_and_preserves_msvc_environment(
    tmp_path, monkeypatch, setup_exit, build_exit
):
    tools = tmp_path / "tools with spaces & punctuation"
    tools.mkdir()
    shutil.copyfile(ci.HERE / "build-x64.cmd", tools / "build-x64.cmd")
    monkeypatch.setattr(ci, "HERE", tools)
    vcvars = tools / "vcvarsall.bat"
    vcvars.write_text(
        "@echo off\nset VSCMD_ARG_TGT_ARCH=x64\nset TEST_MSVC_READY=1\n"
        f"exit /b {setup_exit}\n"
    )
    (tmp_path / "build.py").write_text(
        "import os, pathlib, sys\n"
        "assert os.environ['TEST_MSVC_READY'] == '1'\n"
        "pathlib.Path('built.txt').write_text(' '.join(sys.argv))\n"
        f"sys.exit({build_exit})\n"
    )
    env = dict(os.environ)
    env["VLLM_WINDOWS_VCVARSALL"] = str(vcvars)
    command = ci.build_command(
        "x64", tmp_path, Path(sys.executable), tmp_path, tmp_path / "wheel output", env
    )
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert result.returncode == (setup_exit or build_exit)
    assert (tmp_path / "built.txt").exists() == (setup_exit == 0)


@pytest.mark.skipif(
    not shutil.which("pwsh"), reason="Exercises PowerShell argument passing"
)
@pytest.mark.parametrize("exit_code", [0, 23])
def test_arm64_helper_arguments_and_failures(tmp_path, exit_code):
    env = {**os.environ, **provisioned_environment(tmp_path)}
    env["PATH"] = os.environ["PATH"]
    python, venv, env = ci.build_environment("arm64", env)
    helper = tmp_path / "tools" / "build-win-arm64.ps1"
    helper.parent.mkdir()
    helper.write_text(
        "param($VenvDir, $WheelDir, $CudaPath, $CudaArchList, "
        "$CMakeCudaArchitectures, $MaxJobs)\n"
        "$PSBoundParameters | ConvertTo-Json | Set-Content -LiteralPath "
        "(Join-Path $WheelDir 'arguments.json') -Encoding utf8\n"
        f"exit {exit_code}\n"
    )
    output = tmp_path / "wheel output"
    output.mkdir()
    command = ci.build_command("arm64", tmp_path, python, venv, output, env)
    result = subprocess.run(command, env=env, check=False)
    assert result.returncode == exit_code
    arguments = json.loads((output / "arguments.json").read_text(encoding="utf-8-sig"))
    assert arguments["VenvDir"] == str(venv)
    assert arguments["CudaArchList"] == "12.0+PTX;10.3a"
    assert arguments["CMakeCudaArchitectures"] == "120-real;103-real"
    assert arguments["MaxJobs"] == "8"
