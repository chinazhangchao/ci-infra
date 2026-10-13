import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from buildkite.windows import ci, run_tests


@pytest.mark.parametrize("no_index", [False, True])
@pytest.mark.parametrize("failed_phase", [None, "harness", "wheel", "pytest"])
def test_full_suite_setup_preserves_pool_and_failure(
    tmp_path, monkeypatch, no_index, failed_phase
):
    pool = tmp_path / "pool"
    source = tmp_path / "source"
    output = tmp_path / "artifacts"
    output.mkdir()
    wheel = tmp_path / "vllm.whl"
    original = {
        "PATH": "original",
        "PYTHONPATH": "unrelated-source",
        "PYTEST_ADDOPTS": "-x -k filtered",
        "VLLM_WINDOWS_WHEELHOUSE": str(tmp_path / "wheelhouse with spaces"),
        "VLLM_WINDOWS_NO_INDEX": "1" if no_index else "0",
    }
    calls = []

    def run(args, **kwargs):
        args = [str(arg) for arg in args]
        calls.append((args, kwargs))
        if "venv" in args:
            (Path(args[-1]) / "Lib" / "site-packages").mkdir(parents=True)
        phase = (
            "pytest"
            if str(ci.HERE / "run_tests.py") in args
            else (
                "harness"
                if "-r" in args
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
    )
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
    harness = calls[1][0]
    assert "--only-binary=:all:" in harness
    assert "--find-links" in harness
    assert (
        harness[harness.index("--find-links") + 1]
        == original["VLLM_WINDOWS_WHEELHOUSE"]
    )
    assert ("--no-index" in harness) == no_index
    for command, kwargs in calls[1:]:
        assert command[0] == str(tmp_path / "test-venv" / "Scripts" / "python.exe")
        assert "PYTHONPATH" not in kwargs["env"]
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        assert kwargs["env"]["VIRTUAL_ENV"] == str(tmp_path / "test-venv")
    if failed_phase == "harness":
        assert len(calls) == 2
    if failed_phase == "wheel":
        assert len(calls) == 3
    if not failed_phase or failed_phase == "pytest":
        assert len(calls) == 4
        assert "--no-deps" in calls[2][0]
        assert str(wheel) in calls[2][0]
        assert calls[-1][0][1:3] == ["-I", str(ci.HERE / "run_tests.py")]
        assert calls[-1][1]["cwd"] == source


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


@pytest.mark.parametrize("collection_failure", [False, True])
@pytest.mark.parametrize("test_failure", [False, True])
def test_full_runner_really_runs_optional_tests_without_filters(
    tmp_path, collection_failure, test_failure
):
    python, source, tests, output, env = prepare_full_runner(tmp_path)
    (tests / "test_all.py").write_text(
        "import os, subprocess, sys, pytest, vllm\n"
        "from tests.helper import VALUE\n"
        "def test_installed_package():\n"
        "    assert vllm.ORIGIN == 'built-wheel'\n"
        "    assert VALUE == 17\n"
        "@pytest.mark.optional\n"
        "def test_optional():\n"
        "    assert True\n"
        "def test_child_imports_installed_package():\n"
        "    subprocess.run([sys.executable, '-c', "
        "\"import vllm; assert vllm.ORIGIN == 'built-wheel'\"], check=True)\n"
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
    assert summary["status"] == "failed"
    event = json.loads((output / "pytest-failures.jsonl").read_text())
    assert event["phase"] == "internal"
    assert "broken collection hook" in event["traceback"]
    assert "INTERNAL FAILED:" in result.stdout
