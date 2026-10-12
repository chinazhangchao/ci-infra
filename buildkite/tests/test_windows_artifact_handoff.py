import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from buildkite.tests.test_windows_ci import (
    SHA,
    SOURCE,
    make_wheel,
    provisioned_environment,
)
from buildkite.windows import ci


@pytest.fixture
def built_artifacts(tmp_path, monkeypatch):
    directory = tmp_path / "producer" / "artifacts" / "windows-arm64"
    directory.mkdir(parents=True)
    wheel = make_wheel(directory, "arm64")
    ci.validate_wheel(directory, "arm64")
    env = {
        "BUILDKITE_BUILD_ID": "build-1",
        "BUILDKITE_JOB_ID": "producer-attempt-2",
        "BUILDKITE_STEP_KEY": "windows-cuda-arm64",
        "BUILDKITE_COMMIT": "c" * 40,
    }
    runtime = {
        "platform": "win-arm64",
        "python": "3.13",
        "torch": "2.16.0.dev20261009+cu134",
        "cuda": "13.4",
        "gpu_available": False,
    }
    calls = []
    monkeypatch.setattr(ci, "run", lambda args, **kwargs: calls.append(args))
    ci.publish_build_manifest(directory, "arm64", wheel, SOURCE, runtime, env)
    assert calls[0][:4] == [
        "buildkite-agent",
        "meta-data",
        "set",
        "windows-wheel-arm64",
    ]
    reference = json.loads(calls[0][-1])
    return directory, wheel, env, runtime, reference


def test_build_manifest_binds_wheel_source_runtime_and_producer(built_artifacts):
    directory, wheel, env, runtime, reference = built_artifacts
    verified, manifest = ci.verify_build_artifacts(directory, "arm64", reference, env)
    assert verified == wheel
    assert manifest["source"]["commit"] == SHA
    assert manifest["source"]["ci_infra_commit"] == env["BUILDKITE_COMMIT"]
    assert manifest["runtime"] == runtime
    assert manifest["producer"]["job_id"] == env["BUILDKITE_JOB_ID"]
    assert manifest["wheel"]["sha256"] == ci.sha256_file(wheel)


@pytest.mark.parametrize(
    "problem",
    [
        "manifest-hash",
        "wheel-hash",
        "sidecar",
        "architecture",
        "producer",
        "ci-infra-commit",
        "source",
        "path-traversal",
        "invalid-checksum",
        "duplicate-wheel",
    ],
)
def test_bad_artifacts_fail_before_installation(built_artifacts, problem):
    directory, wheel, env, _runtime, reference = built_artifacts
    manifest_path = directory / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sidecar = directory / f"{wheel.name}.sha256"
    if problem == "manifest-hash":
        reference["manifest_sha256"] = "0" * 64
    elif problem == "wheel-hash":
        wheel.write_bytes(wheel.read_bytes() + b"tampered")
    elif problem == "sidecar":
        sidecar.write_text("0" * 64 + "  " + wheel.name)
    elif problem == "architecture":
        manifest["architecture"] = "x64"
    elif problem == "producer":
        manifest["producer"]["job_id"] = "different-job"
    elif problem == "ci-infra-commit":
        env["BUILDKITE_COMMIT"] = "d" * 40
    elif problem == "source":
        manifest["source"]["commit"] = "HEAD"
    elif problem == "path-traversal":
        manifest["wheel"]["filename"] = "..\\outside.whl"
    elif problem == "invalid-checksum":
        manifest["wheel"]["sha256"] = "not-a-checksum"
    elif problem == "duplicate-wheel":
        shutil.copyfile(wheel, directory / "another-win_arm64.whl")
    if problem not in ("manifest-hash", "wheel-hash", "sidecar", "duplicate-wheel"):
        manifest_path.write_text(json.dumps(manifest))
        reference["manifest_sha256"] = ci.sha256_file(manifest_path)
    sidecar_before = sidecar.read_bytes()
    with pytest.raises(ValueError):
        ci.verify_build_artifacts(directory, "arm64", reference, env)
    assert sidecar.read_bytes() == sidecar_before


@pytest.mark.parametrize("gpu_available", [False, True])
def test_build_only_never_installs_or_runs_tests(tmp_path, monkeypatch, gpu_available):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ci, "sys", SimpleNamespace(platform="win32"))
    env = provisioned_environment(tmp_path)
    env.update(
        BUILDKITE_BUILD_ID="build-1",
        BUILDKITE_JOB_ID="producer",
        BUILDKITE_STEP_KEY="windows-cuda-arm64",
        BUILDKITE_COMMIT="c" * 40,
    )
    prepared = ci.build_environment("arm64", env)
    monkeypatch.setattr(ci, "build_environment", lambda *args: prepared)
    monkeypatch.setattr(ci, "build_command", lambda *args: ["compile-wheel"])
    no_tests = Mock(side_effect=AssertionError("Build-only must not run tests"))
    monkeypatch.setattr(ci, "run_full_tests", no_tests)
    output = tmp_path / "artifacts" / "windows-arm64"
    calls = []

    def run(args, **kwargs):
        args = [str(arg) for arg in args]
        calls.append(args)
        if args[:4] == ["buildkite-agent", "meta-data", "get", ci.SOURCE_KEY]:
            return json.dumps(SOURCE)
        if "-c" in args:
            return json.dumps(
                {
                    "platform": "win-arm64",
                    "python": "3.13",
                    "torch": "test-torch",
                    "cuda": "13.4",
                    "gpu_available": gpu_available,
                }
            )
        if "rev-parse" in args:
            return SHA
        if args == ["compile-wheel"]:
            make_wheel(output, "arm64")
        assert "pip" not in args
        assert str(ci.HERE / "smoke.py") not in args

    monkeypatch.setattr(ci, "run", run)
    ci.build("arm64", "none")
    assert (output / "build-manifest.json").is_file()
    assert len(list(output.glob("*.sha256"))) == 1
    assert calls[-1][:4] == [
        "buildkite-agent",
        "meta-data",
        "set",
        "windows-wheel-arm64",
    ]
    no_tests.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "download",
        "checksum",
        "python",
        "torch",
        "cuda",
        "gpu",
        "tests",
        "wrong-build",
    ],
)
def test_test_job_downloads_exact_producer_and_never_compiles(
    tmp_path, monkeypatch, built_artifacts, failure
):
    directory, wheel, build_env, runtime, reference = built_artifacts
    checkout = tmp_path / "separate-agent-checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    monkeypatch.setattr(ci, "sys", SimpleNamespace(platform="win32"))
    env = {**provisioned_environment(checkout), **build_env}
    env["BUILDKITE_JOB_ID"] = "test-job"
    env["BUILDKITE_STEP_KEY"] = "windows-tests-arm64"
    prepared = ci.build_environment("arm64", env)
    monkeypatch.setattr(ci, "build_environment", lambda *args: prepared)
    monkeypatch.setattr(
        ci, "build_command", Mock(side_effect=AssertionError("Must not build"))
    )
    calls = []
    full_calls = []
    if failure == "wrong-build":
        reference["build_id"] = "another-build"

    def run(args, **kwargs):
        args = [str(arg) for arg in args]
        calls.append(args)
        if args[:3] == ["buildkite-agent", "meta-data", "get"]:
            assert args[3] == "windows-wheel-arm64"
            return json.dumps(reference)
        if args[:3] == ["buildkite-agent", "artifact", "download"]:
            assert args[3] == "artifacts/windows-arm64/*"
            assert args[args.index("--step") + 1] == "producer-attempt-2"
            assert args[args.index("--build") + 1] == "build-1"
            if failure == "download":
                raise subprocess.CalledProcessError(2, args)
            target = Path(args[4]) / "artifacts" / "windows-arm64"
            shutil.copytree(directory, target)
            if failure == "checksum":
                (target / wheel.name).write_bytes(b"corrupt")
        elif "-c" in args:
            current = {**runtime, "gpu_available": failure != "gpu"}
            if failure in ("python", "torch", "cuda"):
                current[failure] = "mismatched-runtime"
            return json.dumps(current)
        elif "rev-parse" in args:
            return SHA
        assert "nvcc.exe" not in " ".join(args)
        assert "build-win-arm64.ps1" not in " ".join(args)

    def full_tests(python, venv, source, downloaded, work, output, test_env):
        full_calls.append(source)
        assert downloaded != wheel
        assert ci.sha256_file(downloaded) == ci.sha256_file(wheel)
        assert output == checkout / "artifacts" / "windows-arm64-tests"
        if failure == "tests":
            raise subprocess.CalledProcessError(1, ["pytest"])

    monkeypatch.setattr(ci, "run", run)
    monkeypatch.setattr(ci, "run_full_tests", full_tests)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            ci.test_build("arm64")
    else:
        ci.test_build("arm64")
        assert len(full_calls) == 1
        assert not full_calls[0].exists()
        fetch = next(args for args in calls if "fetch" in args)
        assert fetch[-1] == SHA
        assert (
            checkout / "artifacts" / "windows-arm64-tests" / "build-manifest.json"
        ).is_file()
        # A retry can land on another agent with no previous checkout or wheel.
        checkout = tmp_path / "retry-agent-checkout"
        checkout.mkdir()
        monkeypatch.chdir(checkout)
        prepared[2]["VLLM_WINDOWS_WORK_ROOT"] = str(checkout / "work")
        ci.test_build("arm64")
        assert len(full_calls) == 2
        assert full_calls[0] != full_calls[1]
        assert not full_calls[1].exists()
        downloads = [
            args
            for args in calls
            if args[:3] == ["buildkite-agent", "artifact", "download"]
        ]
        assert len(downloads) == 2
        assert all(
            args[args.index("--step") + 1] == reference["job_id"] for args in downloads
        )
        assert not any(
            args[:3] == ["buildkite-agent", "meta-data", "set"] for args in calls
        )
    if failure not in (None, "tests"):
        assert not full_calls
        assert not any("fetch" in args for args in calls)


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (
            ["build", "--architecture", "arm64", "--test-suite", "none"],
            ("arm64", "none"),
        ),
        (["test", "--architecture", "arm64"], ("arm64",)),
        (["build", "--architecture", "x64"], ("x64", "smoke")),
    ],
)
def test_cli_dispatches_independent_jobs(monkeypatch, arguments, expected):
    build = Mock()
    test = Mock()
    monkeypatch.setattr(ci, "build", build)
    monkeypatch.setattr(ci, "test_build", test)
    monkeypatch.setattr(ci.sys, "argv", ["ci.py", *arguments])
    ci.main()
    (test if arguments[0] == "test" else build).assert_called_once_with(*expected)
    (build if arguments[0] == "test" else test).assert_not_called()
