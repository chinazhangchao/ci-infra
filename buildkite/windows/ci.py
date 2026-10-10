"""Native Windows CUDA builds using provisioned private Buildkite agents."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPOSITORY = "https://github.com/vortex-captain/vllm-windows.git"
DEFAULT_BRANCH = "v029_win_arm64"
SOURCE_KEY = "windows-vllm-source"
HERE = Path(__file__).resolve().parent
PLATFORMS = {"x64": "win_amd64", "arm64": "win_arm64"}
ARM64_OPTIONS = {
    "VLLM_WINDOWS_VS_PATH": "VisualStudioPath",
    "VLLM_WINDOWS_MSVC_VERSION": "MsvcToolsetVersion",
    "VLLM_WINDOWS_RUST_VS_PATH": "RustVisualStudioPath",
    "VLLM_WINDOWS_RUST_MSVC_VERSION": "RustMsvcToolsetVersion",
    "VLLM_WINDOWS_SDK_VERSION": "WindowsSdkVersion",
    "VLLM_WINDOWS_PERL_PATH": "PerlPath",
    "VLLM_WINDOWS_PROTOC_PATH": "ProtocPath",
    "VLLM_WINDOWS_PROTOC_INCLUDE_PATH": "ProtocIncludePath",
    "VLLM_WINDOWS_VERSION_OVERRIDE": "VersionOverride",
}


def run(args, *, cwd=None, env=None, capture=False):
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        check=True,
    )
    return result.stdout.strip() if capture else None


def required(env, name):
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be configured on the private Windows agent.")
    return value


def validate_source(source):
    if (
        not isinstance(source, dict)
        or source.get("repository") != REPOSITORY
        or not isinstance(source.get("commit"), str)
        or not re.fullmatch(r"[0-9a-fA-F]{40}", source["commit"])
    ):
        raise ValueError(
            "Invalid Windows source metadata: expected repository and SHA."
        )
    return source


def resolve_source():
    # Keep retries of the resolver and both architecture jobs on the same SHA.
    exists = subprocess.run(
        ["buildkite-agent", "meta-data", "exists", SOURCE_KEY], check=False
    )
    if exists.returncode == 0:
        source = validate_source(
            json.loads(
                run(["buildkite-agent", "meta-data", "get", SOURCE_KEY], capture=True)
            )
        )
    elif exists.returncode == 100:
        commit = os.environ.get("VLLM_WINDOWS_COMMIT", "").strip()
        branch = os.environ.get("VLLM_WINDOWS_BRANCH", DEFAULT_BRANCH)
        if not commit:
            ref = f"refs/heads/{branch}"
            run(["git", "check-ref-format", ref])
            refs = run(
                ["git", "ls-remote", "--exit-code", "--refs", REPOSITORY, ref],
                capture=True,
            )
            matches = [line.split() for line in refs.splitlines()]
            if len(matches) != 1 or len(matches[0]) != 2 or matches[0][1] != ref:
                raise ValueError(f"Expected exactly one branch ref for {ref}.")
            commit = matches[0][0]
        source = validate_source(
            {"repository": REPOSITORY, "branch": branch, "commit": commit}
        )
        run(["buildkite-agent", "meta-data", "set", SOURCE_KEY, json.dumps(source)])
    else:
        raise subprocess.CalledProcessError(exists.returncode, exists.args)
    print(json.dumps(source, indent=2), flush=True)


def build_environment(architecture, env):
    env = dict(env)
    venv = Path(required(env, "VLLM_WINDOWS_VENV")).resolve()
    python = next(
        (
            path
            for path in (venv / "Scripts" / "python.exe", venv / "python.exe")
            if path.is_file()
        ),
        None,
    )
    if python is None:
        raise ValueError(f"No provisioned Python interpreter found in {venv}.")
    cuda = Path(required(env, "CUDA_PATH")).resolve()
    for relative in (
        Path("bin") / "nvcc.exe",
        Path("lib") / architecture / "cudart.lib",
    ):
        if not (cuda / relative).is_file():
            raise ValueError(
                f"Required {architecture} CUDA file is missing: {cuda / relative}"
            )
    required(env, "TORCH_CUDA_ARCH_LIST")
    jobs = int(env.get("MAX_JOBS", "8"))
    if not 1 <= jobs <= 256:
        raise ValueError("MAX_JOBS must be between 1 and 256.")
    env.update(
        CUDA_PATH=str(cuda),
        CUDA_HOME=str(cuda),
        CUDA_ROOT=str(cuda),
        DISTUTILS_USE_SDK="1",
        VLLM_TARGET_DEVICE="cuda",
        VLLM_USE_PRECOMPILED="0",
        VLLM_USE_PRECOMPILED_RUST="0",
        VLLM_REQUIRE_RUST_FRONTEND="1",
        MAX_JOBS=str(jobs),
        CMAKE_BUILD_PARALLEL_LEVEL=str(jobs),
    )
    env.pop("VLLM_PRECOMPILED_WHEEL_LOCATION", None)
    env.pop("VLLM_BUILD_BASE", None)
    env["PATH"] = os.pathsep.join(
        [
            str(python.parent),
            str(venv / "Library" / "bin"),
            str(cuda / "bin"),
            env["PATH"],
        ]
    )
    if architecture == "x64":
        vcvars = Path(required(env, "VLLM_WINDOWS_VCVARSALL")).resolve()
        if not vcvars.is_file():
            raise ValueError(f"vcvarsall.bat was not found: {vcvars}")
        env["VLLM_WINDOWS_VCVARSALL"] = str(vcvars)
    else:
        required(env, "CMAKE_CUDA_ARCHITECTURES")
    return python, venv, env


def build_command(architecture, source, python, venv, output, env):
    if architecture == "x64":
        env["VLLM_WINDOWS_BUILD_PYTHON"] = str(python)
        env["VLLM_WINDOWS_WHEEL_DIR"] = str(output)
        # CALL keeps cmd.exe from stripping the wrapper path's outer quotes.
        return [env["COMSPEC"], "/d", "/c", "call", str(HERE / "build-x64.cmd")]
    helper = source / "tools" / "build-win-arm64.ps1"
    if not helper.is_file():
        raise ValueError(
            f"The selected fork commit has no ARM64 build helper: {helper}"
        )
    command = [
        "pwsh",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-File",
        str(helper),
        "-VenvDir",
        str(venv),
        "-WheelDir",
        str(output),
        "-CudaPath",
        env["CUDA_PATH"],
        "-CudaArchList",
        env["TORCH_CUDA_ARCH_LIST"],
        "-CMakeCudaArchitectures",
        env["CMAKE_CUDA_ARCHITECTURES"],
        "-MaxJobs",
        env["MAX_JOBS"],
    ]
    for variable, parameter in ARM64_OPTIONS.items():
        if env.get(variable):
            command.extend([f"-{parameter}", env[variable]])
    return command


def validate_wheel(output, architecture):
    wheels = list(output.glob("*.whl"))
    platform = PLATFORMS[architecture]
    if len(wheels) != 1 or not wheels[0].name.endswith(f"-{platform}.whl"):
        raise ValueError(f"Expected exactly one {platform} wheel, got: {wheels}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        metadata_files = [name for name in members if name.endswith(".dist-info/WHEEL")]
        if len(metadata_files) != 1:
            raise ValueError("Wheel must contain exactly one WHEEL metadata file.")
        tags = [
            line.removeprefix("Tag: ").strip()
            for line in archive.read(metadata_files[0]).decode().splitlines()
            if line.startswith("Tag: ")
        ]
        if not tags or any(tag.rsplit("-", 1)[-1] != platform for tag in tags):
            raise ValueError(f"Wheel metadata does not target {platform}: {tags}")
        if not any(re.fullmatch(r"vllm/_C(?:\.[^/]+)?\.pyd", name) for name in members):
            raise ValueError("Wheel does not contain the compiled vllm._C extension.")
    with wheel.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    (output / f"{wheel.name}.sha256").write_text(
        f"{digest.hexdigest()}  {wheel.name}\n", encoding="utf-8"
    )
    return wheel


def build(architecture):
    if sys.platform != "win32":
        raise RuntimeError("Windows CUDA builds must run on Windows agents.")
    python, venv, env = build_environment(architecture, os.environ)
    source_info = validate_source(
        json.loads(
            run(["buildkite-agent", "meta-data", "get", SOURCE_KEY], capture=True)
        )
    )
    output = Path.cwd() / "artifacts" / f"windows-{architecture}"
    # A dirty output directory must never turn an old wheel into a passing build.
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.json").write_text(
        json.dumps(
            {**source_info, "ci_infra_commit": env.get("BUILDKITE_COMMIT")},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    probe = run(
        [
            python,
            "-I",
            "-c",
            (
                "import json, sysconfig, torch, build; "
                "print(json.dumps(dict(platform=sysconfig.get_platform(), "
                "torch=torch.__version__, cuda=torch.version.cuda, "
                "gpu_available=torch.cuda.is_available())))"
            ),
        ],
        env=env,
        capture=True,
    )
    runtime = json.loads(probe)
    if runtime["platform"].replace("-", "_") != PLATFORMS[architecture]:
        raise ValueError(
            f"The provisioned Python does not target {architecture}: {runtime}"
        )
    if not runtime["cuda"] or not runtime["gpu_available"]:
        raise ValueError(
            f"A CUDA-enabled PyTorch and a working NVIDIA GPU are required: {runtime}"
        )
    if architecture == "arm64" and runtime["cuda"] != "13.4":
        raise ValueError(
            "The fork's ARM64 helper requires PyTorch built for CUDA 13.4."
        )
    (output / "runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    run([Path(env["CUDA_PATH"]) / "bin" / "nvcc.exe", "--version"], env=env)

    work_root = Path(required(env, "VLLM_WINDOWS_WORK_ROOT")).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vllm-", dir=work_root) as work:
        source = Path(work) / "src"
        source.mkdir()
        run(["git", "init", str(source)])
        run(["git", "-C", source, "config", "core.longpaths", "true"])
        run(["git", "-C", source, "remote", "add", "origin", REPOSITORY])
        # setuptools-scm needs tags and history for the x64 wheel version.
        run(["git", "-C", source, "fetch", "--tags", "origin", source_info["commit"]])
        run(["git", "-C", source, "checkout", "--detach", source_info["commit"]])
        actual_commit = run(["git", "-C", source, "rev-parse", "HEAD"], capture=True)
        if actual_commit.lower() != source_info["commit"].lower():
            raise ValueError(
                f"Checked out an unexpected source commit: {actual_commit}"
            )
        run(["git", "-C", source, "submodule", "update", "--init", "--recursive"])
        command = build_command(architecture, source, python, venv, output, env)
        run(command, cwd=source, env=env)
        wheel = validate_wheel(output, architecture)
        installed = Path(work) / "installed"
        # Do not replace vLLM or its dependencies in the pool's provisioned environment.
        run(
            [
                python,
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-deps",
                "--no-compile",
                "--ignore-installed",
                "--target",
                installed,
                wheel,
            ],
            cwd=work,
            env=env,
        )
        run(
            [
                python,
                "-I",
                HERE / "smoke.py",
                "--package-dir",
                installed,
                "--architecture",
                architecture,
                "--output",
                output / "smoke.json",
            ],
            cwd=work,
            env=env,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("resolve")
    build_parser = subcommands.add_parser("build")
    build_parser.add_argument("--architecture", choices=PLATFORMS, required=True)
    args = parser.parse_args()
    if args.command == "resolve":
        resolve_source()
    else:
        build(args.architecture)


if __name__ == "__main__":
    main()
