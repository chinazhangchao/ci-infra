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
        run_tests, "sys", SimpleNamespace(prefix=str(tmp_path / "venv"), path=[])
    )
    importer = Mock(
        return_value=SimpleNamespace(__file__=str(source / "vllm" / "__init__.py"))
    )
    monkeypatch.setattr(run_tests, "importlib", SimpleNamespace(import_module=importer))
    with pytest.raises(RuntimeError, match="job's installed wheel"):
        run_tests.run_tests(source, tmp_path)
    importer.assert_called_once_with("vllm")


@pytest.mark.parametrize("collection_failure", [False, True])
@pytest.mark.parametrize("test_failure", [False, True])
def test_full_runner_really_runs_optional_tests_without_filters(
    tmp_path, collection_failure, test_failure
):
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
            "def test_a_failure():\n    assert False\n"
            "def test_b_runs_after_failure():\n    assert True\n"
        )
    output = tmp_path / "artifacts"
    output.mkdir()
    env = dict(os.environ)
    env["PYTEST_ADDOPTS"] = "-x -k nonexistent"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
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
    assert (output / "pytest.xml").is_file()
