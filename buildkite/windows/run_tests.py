"""Run a selected fork pytest suite against the installed Windows wheel."""

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


FATAL_CUDA_ERRORS = (
    "torch.OutOfMemoryError",
    "CUDA error: out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "CUDA error: an illegal memory access",
    "CUDA error: unspecified launch failure",
    "CUDA_ERROR_LAUNCH_FAILED",
    "device-side assert triggered",
    "Sticky error detected",
)
CHECKPOINT_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0)


def _core_test_targets():
    tests = Path("tests")
    kernels = tests / "kernels" / "core"
    targets = [
        str(tests / "utils_" / name)
        for name in (
            "test_import_utils.py",
            "test_collection_utils.py",
            "test_cache.py",
        )
    ]
    # Explicit parameter IDs keep large tensors and additional devices out.
    for dtype in range(3):
        for tokens in (7, 83):
            for activation in ("silu_and_mul", "mul_and_silu", "gelu", "gelu_tanh"):
                targets.append(
                    f"{kernels / 'test_activation.py'}::test_act_and_mul"
                    f"[cuda:0-0-dtype{dtype}-512-{tokens}-{activation}]"
                )
    for strided in (False, True):
        for dtype in range(3):
            for residual in (False, True):
                for hidden in (768, 769):
                    for tokens in (7, 83):
                        targets.append(
                            f"{kernels / 'test_layernorm.py'}::test_rms_norm"
                            f"[{strided}-cuda:0-0-dtype{dtype}-{residual}-{hidden}-{tokens}]"
                        )
    targets.append(
        f"{kernels / 'test_layernorm.py'}::test_gemma_rms_norm_mixed_input_weight_dtype"
    )
    for dtype in range(2):
        for residual in (False, True):
            targets.append(
                f"{kernels / 'test_layernorm.py'}::test_fused_rms_norm_quant"
                f"[False-cuda:0-0-1.0-dtype{dtype}-{residual}-768-7]"
            )
    for use_key in (False, True):
        for dtype in range(2):
            for rotary_dim in (None, 32):
                for layout in ("_get_batch_tensor_shape", "_get_padded_tensor_shape"):
                    for neox in (False, True):
                        targets.append(
                            f"{kernels / 'test_pos_encoding.py'}::test_rotary_embedding"
                            f"[{use_key}-cuda:0-0-dtype{dtype}-{rotary_dim}-64-17-11-5-{layout}-{neox}]"
                        )
    cache = tests / "kernels" / "attention" / "test_cache.py"
    for cache_dtype in ("auto", "fp8"):
        for dtype in range(2):
            for head_size in (64, 80):
                targets.append(
                    f"{cache}::test_reshape_and_cache"
                    f"[{cache_dtype}-cuda:0-0-dtype{dtype}-1024-16-{head_size}-8-42]"
                )
    return tuple(targets)


TEST_TARGETS = {
    "core": _core_test_targets(),
    "full": ("tests",),
    "deepgemm": (
        str(Path("tests") / "kernels" / "moe" / "test_silu_mul_fp8_quant_deep_gemm.py"),
    ),
}


class TestResults:
    def __init__(self, output, source, test_suite="full"):
        if test_suite not in TEST_TARGETS:
            raise ValueError(f"Unsupported pytest suite: {test_suite!r}")
        self.output = output
        self.terminal = None
        self.summary = {
            "suite": test_suite,
            "selected_paths": list(TEST_TARGETS[test_suite]),
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
        destination = self.output / "pytest-summary.json"
        # Windows readers/scanners can temporarily prevent an atomic rename.
        for attempt in range(len(CHECKPOINT_RETRY_DELAYS) + 1):
            try:
                temporary.replace(destination)
                return
            except OSError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33):
                    raise
                if attempt == len(CHECKPOINT_RETRY_DELAYS):
                    error.add_note(
                        f"Checkpoint replacement retries exhausted. Previous checkpoint: "
                        f"{destination}; pending checkpoint: {temporary}."
                    )
                    raise
                if attempt == 0:
                    print(
                        f"WARNING: Cannot replace pytest checkpoint (WinError "
                        f"{error.winerror}); retrying without removing {destination}.",
                        file=sys.stderr,
                        flush=True,
                    )
                time.sleep(CHECKPOINT_RETRY_DELAYS[attempt])

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

    test_suite = results.summary["suite"]
    targets = []
    for selection in TEST_TARGETS[test_suite]:
        filename, separator, node = selection.partition("::")
        target = source / filename
        if test_suite != "full" and not target.is_file():
            raise FileNotFoundError(
                f"The selected vLLM commit has no {test_suite} test target: {target}"
            )
        targets.append(f"{target}{separator}{node}")
    if test_suite == "deepgemm":
        # Surface the failing kernel at its launch, not at a later synchronization.
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    results.update(cuda_launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"))

    vllm = importlib.import_module("vllm")
    package = Path(vllm.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()) or package.is_relative_to(
        source
    ):
        raise RuntimeError("Tests must import vLLM from the job's installed wheel.")

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
        *targets,
        "--optional",
        "--continue-on-collection-errors",
        "--maxfail=0",
        "--import-mode=append",
        "-o",
        "addopts=",
        "-o",
        "pythonpath=",
        "--tb=long" if test_suite == "deepgemm" else "--tb=short",
        "-ra",
        f"--junitxml={output / 'pytest.xml'}",
    ]
    if test_suite != "full":
        args.append("-vv")
    results.update(vllm_package=str(package), arguments=args)
    return int(pytest.main(args, plugins=[results]))


def run_tests(source, output, test_suite="full"):
    source = source.resolve()
    output = output.resolve()
    results = TestResults(output, source, test_suite)
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
                else "error"
                if exit_code == 3
                else "failed"
            )
            results.update(status=status, exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-suite", choices=TEST_TARGETS, default="full")
    args = parser.parse_args()
    return run_tests(args.source_dir, args.output, args.test_suite)


if __name__ == "__main__":
    raise SystemExit(main())
