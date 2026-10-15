"""Run the fork's entire pytest tree against the installed Windows wheel."""

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path


FATAL_CUDA_ERRORS = (
    "torch.OutOfMemoryError",
    "CUDA error: out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "CUDA error: an illegal memory access",
    "device-side assert triggered",
    "Sticky error detected",
)


class TestResults:
    def __init__(self, output, source):
        self.output = output
        self.terminal = None
        self.summary = {
            "suite": "full",
            "status": "initializing",
            "source": str(source),
            "exit_code": None,
            "active_test": None,
            "collected": 0,
            "deselected": 0,
            "completed": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "collection_errors": 0,
            "stop_reason": None,
            "windows_test_helpers": [],
        }
        self.update()

    def update(self, **fields):
        self.summary.update(fields)
        # Keep a readable checkpoint even if the process dies during a write.
        temporary = self.output / "pytest-summary.json.tmp"
        temporary.write_text(
            json.dumps(self.summary, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(self.output / "pytest-summary.json")

    def failure(self, nodeid, phase, details, sections=()):
        event = {
            "nodeid": nodeid,
            "phase": phase,
            "traceback": details,
            "sections": list(sections),
        }
        with (self.output / "pytest-failures.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(event) + "\n")
        title = f"{phase.upper()} FAILED: {nodeid or 'pytest runner'}"
        if self.terminal is not None:
            self.terminal.write_sep("=", title, red=True)
            self.terminal.write_line(details)
            for name, content in sections:
                self.terminal.write_sep("-", name)
                self.terminal.write_line(content)
        else:
            print(f"\n{title}\n{details}", file=sys.stderr, flush=True)
        sys.stdout.flush()

    def pytest_sessionstart(self, session):
        self.terminal = session.config.pluginmanager.get_plugin("terminalreporter")
        self.update(status="collecting")

    def pytest_collection_finish(self, session):
        compatibility = sys.modules.get("_vllm_ci_windows_helpers")
        self.update(
            collected=len(session.items),
            status="running",
            windows_test_helpers=list(getattr(compatibility, "APPLIED_HELPERS", [])),
        )

    def pytest_deselected(self, items):
        self.update(deselected=self.summary["deselected"] + len(items))

    def pytest_collectreport(self, report):
        if report.failed:
            self.failure(
                report.nodeid, "collection", report.longreprtext, report.sections
            )
            self.update(collection_errors=self.summary["collection_errors"] + 1)

    def pytest_runtest_logstart(self, nodeid, location):
        self.update(active_test=nodeid)

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failure(
                report.nodeid, report.when, report.longreprtext, report.sections
            )
            self.update(failed=self.summary["failed"] + 1)
            if any(error in report.longreprtext for error in FATAL_CUDA_ERRORS):
                self.update(
                    stop_reason=(
                        f"CUDA resource/context failure in {report.nodeid} "
                        f"({report.when}); remaining tests were not run."
                    )
                )
        elif report.skipped:
            self.update(skipped=self.summary["skipped"] + 1)
        elif report.when == "call" and report.passed:
            self.update(passed=self.summary["passed"] + 1)

    def pytest_runtest_logfinish(self, nodeid, location):
        self.update(active_test=None, completed=self.summary["completed"] + 1)
        if self.summary["stop_reason"]:
            # Abort after teardown so a bad CUDA context cannot contaminate later tests.
            pytest = importlib.import_module("pytest")
            pytest.exit(self.summary["stop_reason"], returncode=1)

    def pytest_internalerror(self, excrepr, excinfo):
        self.failure(None, "internal", str(excrepr))
        self.update(status="error")

    def pytest_keyboard_interrupt(self, excinfo):
        self.failure(self.summary["active_test"], "interrupted", str(excinfo.getrepr()))
        self.update(status="interrupted")


def execute_tests(source, output, results):
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
    os.environ["_VLLM_CI_TEST_SOURCE"] = str(source)
    os.environ.pop("PYTEST_ADDOPTS", None)
    # The checkout's tests package must also beat unrelated site-packages/tests.
    sys.path[:0] = [str(site_packages), str(source)]
    pytest = importlib.import_module("pytest")
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
        "--tb=short",
        "-ra",
        f"--junitxml={output / 'pytest.xml'}",
    ]
    results.update(vllm_package=str(package), arguments=args)
    return int(pytest.main(args, plugins=[results]))


def run_tests(source, output):
    source = source.resolve()
    output = output.resolve()
    results = TestResults(output, source)
    exit_code = None
    try:
        exit_code = execute_tests(source, output, results)
        return exit_code
    finally:
        if exit_code is None:
            results.failure(None, "runner", traceback.format_exc())
            results.update(status="error")
        else:
            status = (
                "aborted"
                if results.summary["stop_reason"]
                else "passed"
                if exit_code == 0
                else "interrupted"
                if exit_code == 2
                else "failed"
            )
            results.update(status=status, exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run_tests(args.source_dir, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
