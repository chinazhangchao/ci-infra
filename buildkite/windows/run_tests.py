"""Run the fork's entire pytest tree against the installed Windows wheel."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


class TestResults:
    def __init__(self):
        self.collected = 0
        self.deselected = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.collection_errors = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors += 1

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1
        elif report.when == "call" and report.passed:
            self.passed += 1


def run_tests(source, output):
    source = source.resolve()
    output = output.resolve()
    tests = source / "tests"
    if not tests.is_dir():
        raise FileNotFoundError(
            f"The selected vLLM commit has no tests directory: {tests}"
        )

    vllm = importlib.import_module("vllm")
    package = Path(vllm.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()) or package.is_relative_to(
        source
    ):
        raise RuntimeError(
            "Full tests must import vLLM from the job's installed wheel."
        )

    # Keep installed packages ahead of the checkout, including in spawned Python
    # workers and `python -m vllm...` commands launched by the tests.
    site_packages = package.parent.parent
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site_packages), str(source)])
    os.environ["PYTHONSAFEPATH"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    # The checkout's tests package must also beat unrelated site-packages/tests.
    sys.path[:0] = [str(site_packages), str(source)]
    pytest = importlib.import_module("pytest")
    results = TestResults()
    args = [
        str(tests),
        "--optional",
        "--continue-on-collection-errors",
        "--maxfail=0",
        "--import-mode=append",
        "-o",
        "addopts=",
        "-o",
        "pythonpath=",
        "-ra",
        f"--junitxml={output / 'pytest.xml'}",
    ]
    exit_code = int(pytest.main(args, plugins=[results]))
    summary = {
        "suite": "full",
        "status": "passed" if exit_code == 0 else "failed",
        "source": str(source),
        "vllm_package": str(package),
        "arguments": args,
        "exit_code": exit_code,
        **vars(results),
    }
    (output / "pytest-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run_tests(args.source_dir, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
