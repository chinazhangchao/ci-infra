"""Native Windows CUDA builds using provisioned private Buildkite agents."""

import argparse
import hashlib
import json
import os
import re
import shutil
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
TEST_BUILD_ID_ENV = "VLLM_WINDOWS_TEST_BUILD_ID"
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
    # CMake-created clones and nested submodules do not inherit local Git config.
    git_config_count = env.get("GIT_CONFIG_COUNT", "0")
    if not re.fullmatch(r"[0-9]+", git_config_count):
        raise ValueError("GIT_CONFIG_COUNT must be a non-negative integer.")
    git_config_count = int(git_config_count)
    env[f"GIT_CONFIG_KEY_{git_config_count}"] = "core.longpaths"
    env[f"GIT_CONFIG_VALUE_{git_config_count}"] = "true"
    env["GIT_CONFIG_COUNT"] = str(git_config_count + 1)
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


def git_longpaths_environment(env, work):
    prepared = dict(env)
    if "GIT_CONFIG_GLOBAL" in env:
        includes = [Path(env["GIT_CONFIG_GLOBAL"])] if env["GIT_CONFIG_GLOBAL"] else []
    else:
        home = Path(env.get("HOME") or env.get("USERPROFILE") or Path.home())
        xdg = Path(env.get("XDG_CONFIG_HOME") or home / ".config")
        includes = [xdg / "git" / "config", home / ".gitconfig"]
    config = work / "git-longpaths.config"
    content = ""
    if includes:
        content = "[include]\n" + "".join(
            f"    path = {json.dumps(str(path.absolute()), ensure_ascii=False)}\n"
            for path in includes
        )
    content += "[core]\n    longpaths = true\n"
    with config.open("x", encoding="utf-8") as stream:
        stream.write(content)
    prepared["GIT_CONFIG_GLOBAL"] = str(config)
    return prepared


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_wheel(output, architecture, *, write_checksum=True):
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
        if not any(
            re.fullmatch(r"vllm/_C(?:_stable_libtorch)?(?:\.[^/]+)?\.pyd", name)
            for name in members
        ):
            raise ValueError(
                "Wheel does not contain a compiled vLLM core extension "
                "(_C or _C_stable_libtorch)."
            )
    if write_checksum:
        (output / f"{wheel.name}.sha256").write_text(
            f"{sha256_file(wheel)}  {wheel.name}\n", encoding="utf-8"
        )
    return wheel


def probe_runtime(python, env, architecture, *, require_gpu=True):
    runtime = json.loads(
        run(
            [
                python,
                "-I",
                "-c",
                (
                    "import json, sys, sysconfig, torch, build; "
                    "print(json.dumps(dict(platform=sysconfig.get_platform(), "
                    "python=f'{sys.version_info.major}.{sys.version_info.minor}', "
                    "torch=torch.__version__, cuda=torch.version.cuda, "
                    "gpu_available=torch.cuda.is_available())))"
                ),
            ],
            env=env,
            capture=True,
        )
    )
    if runtime["platform"].replace("-", "_") != PLATFORMS[architecture]:
        raise ValueError(
            f"The provisioned Python does not target {architecture}: {runtime}"
        )
    if not runtime["cuda"] or (require_gpu and not runtime["gpu_available"]):
        raise ValueError(
            f"CUDA-enabled PyTorch{' and a working NVIDIA GPU' if require_gpu else ''} required: {runtime}"
        )
    if architecture == "arm64" and runtime["cuda"] != "13.4":
        raise ValueError(
            "The fork's ARM64 helper requires PyTorch built for CUDA 13.4."
        )
    return runtime


def checkout_source(source, source_info, env):
    source.mkdir()
    run(["git", "init", str(source)], env=env)
    run(["git", "-C", source, "config", "core.longpaths", "true"], env=env)
    run(["git", "-C", source, "remote", "add", "origin", REPOSITORY], env=env)
    # setuptools-scm needs tags and history for the x64 wheel version.
    run(
        ["git", "-C", source, "fetch", "--tags", "origin", source_info["commit"]],
        env=env,
    )
    run(["git", "-C", source, "checkout", "--detach", source_info["commit"]], env=env)
    actual_commit = run(
        ["git", "-C", source, "rev-parse", "HEAD"], env=env, capture=True
    )
    if actual_commit.lower() != source_info["commit"].lower():
        raise ValueError(f"Checked out an unexpected source commit: {actual_commit}")
    run(["git", "-C", source, "submodule", "update", "--init", "--recursive"], env=env)


def publish_build_manifest(output, architecture, wheel, source_info, runtime, env):
    producer = {
        "build_id": required(env, "BUILDKITE_BUILD_ID"),
        "job_id": required(env, "BUILDKITE_JOB_ID"),
        "step_key": required(env, "BUILDKITE_STEP_KEY"),
    }
    manifest = {
        "schema_version": 1,
        "architecture": architecture,
        "producer": producer,
        "source": {**source_info, "ci_infra_commit": required(env, "BUILDKITE_COMMIT")},
        "runtime": runtime,
        "wheel": {"filename": wheel.name, "sha256": sha256_file(wheel)},
    }
    manifest_path = output / "build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # The test job depends on successful completion, including automatic artifact
    # upload. Pin the producer attempt so test retries cannot mix jobs' files.
    reference = {**producer, "manifest_sha256": sha256_file(manifest_path)}
    run(
        [
            "buildkite-agent",
            "meta-data",
            "set",
            f"windows-wheel-{architecture}",
            json.dumps(reference),
        ],
        env=env,
    )


def verify_build_artifacts(
    directory, architecture, reference, env, *, allow_updated_runner=False
):
    manifest_path = directory / "build-manifest.json"
    if sha256_file(manifest_path) != reference["manifest_sha256"]:
        raise ValueError("Build manifest checksum does not match the producing job.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("architecture") != architecture
    ):
        raise ValueError("Unsupported build manifest or wrong wheel architecture.")
    producer = {key: reference[key] for key in ("build_id", "job_id", "step_key")}
    if manifest.get("producer") != producer:
        raise ValueError("Build manifest identifies a different producing job.")
    source_info = validate_source(manifest["source"])
    build_commit = source_info.get("ci_infra_commit")
    if not isinstance(build_commit, str) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", build_commit
    ):
        raise ValueError("Build manifest has no valid ci-infra commit.")
    if not allow_updated_runner and build_commit != required(env, "BUILDKITE_COMMIT"):
        raise ValueError("Build and test jobs must use the same ci-infra commit.")
    wheel_info = manifest["wheel"]
    filename = wheel_info["filename"]
    if not isinstance(filename, str) or not re.fullmatch(
        r"[A-Za-z0-9_.+!-]+\.whl", filename
    ):
        raise ValueError("Invalid wheel filename in build manifest.")
    wheel = directory / filename
    checksum = wheel_info["sha256"]
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Invalid wheel checksum in build manifest.")
    if sha256_file(wheel) != checksum:
        raise ValueError("Wheel checksum does not match the producing job.")
    expected = f"{checksum}  {filename}"
    if (directory / f"{filename}.sha256").read_text(
        encoding="utf-8"
    ).strip() != expected:
        raise ValueError("Wheel checksum sidecar disagrees with the build manifest.")
    if validate_wheel(directory, architecture, write_checksum=False) != wheel:
        raise ValueError("Unexpected wheel in downloaded build artifacts.")
    return wheel, manifest


def test_build(architecture, source_build_id=None, test_suite="full"):
    if test_suite not in ("full", "deepgemm"):
        raise ValueError(f"Unsupported pytest suite: {test_suite!r}")
    if sys.platform != "win32":
        raise RuntimeError("Windows CUDA tests must run on Windows agents.")
    python, venv, env = build_environment(architecture, os.environ)
    if source_build_id is None:
        source_build_id = env.get(TEST_BUILD_ID_ENV)
    if source_build_id and not re.fullmatch(
        r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", source_build_id
    ):
        raise ValueError(
            f"{TEST_BUILD_ID_ENV}/--source-build-id must be the producer's "
            "BUILDKITE_BUILD_ID UUID, not its build number or URL."
        )
    requested_build = source_build_id or required(env, "BUILDKITE_BUILD_ID")
    metadata_args = [
        "buildkite-agent",
        "meta-data",
        "get",
        f"windows-wheel-{architecture}",
    ]
    if source_build_id:
        metadata_args += ["--build", source_build_id]
        print(
            f"Testing wheel artifacts from Buildkite build {source_build_id}; no compilation.",
            flush=True,
        )
    reference = json.loads(
        run(
            metadata_args,
            env=env,
            capture=True,
        )
    )
    if (
        not isinstance(reference, dict)
        or reference.get("build_id") != requested_build
        or reference.get("step_key") != f"windows-cuda-{architecture}"
        or not isinstance(reference.get("job_id"), str)
        or not reference["job_id"]
        or not isinstance(reference.get("manifest_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", reference["manifest_sha256"])
    ):
        raise ValueError(
            f"Invalid Windows wheel producer reference for Buildkite build {requested_build}."
        )
    output = Path.cwd() / "artifacts" / f"windows-{architecture}-tests"
    output.mkdir(parents=True, exist_ok=False)
    (output / "producer.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    (output / "test-run.json").write_text(
        json.dumps(
            {
                "build_id": required(env, "BUILDKITE_BUILD_ID"),
                "ci_infra_commit": required(env, "BUILDKITE_COMMIT"),
                "source_build_id": requested_build,
                "suite": test_suite,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    work_root = Path(required(env, "VLLM_WINDOWS_WORK_ROOT")).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vllm-test-", dir=work_root) as work:
        work = Path(work)
        env = git_longpaths_environment(env, work)
        incoming = work / "download"
        incoming.mkdir()
        run(
            [
                "buildkite-agent",
                "artifact",
                "download",
                f"artifacts/windows-{architecture}/*",
                str(incoming),
                "--build",
                reference["build_id"],
                "--step",
                reference["job_id"],
            ],
            env=env,
        )
        wheel, manifest = verify_build_artifacts(
            incoming / "artifacts" / f"windows-{architecture}",
            architecture,
            reference,
            env,
            allow_updated_runner=bool(source_build_id),
        )
        (output / "build-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        runtime = probe_runtime(python, env, architecture)
        for field in ("platform", "python", "torch", "cuda"):
            expected = manifest["runtime"].get(field)
            if not expected or runtime.get(field) != expected:
                raise ValueError(
                    f"Test runtime {field}={runtime.get(field)!r} does not match build {expected!r}."
                )
        (output / "runtime.json").write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
        )
        source = work / "src"
        checkout_source(source, manifest["source"], env)
        run_full_tests(
            python, venv, source, wheel, work, output, env, architecture, test_suite
        )


def test_compiler_environment(architecture, env):
    prepared = {key.upper(): value for key, value in env.items()}
    variable = (
        "VLLM_WINDOWS_VS_PATH" if architecture == "arm64" else "VLLM_WINDOWS_VCVARSALL"
    )
    vcvars = Path(required(prepared, variable)).resolve()
    if vcvars.is_dir():
        vcvars = vcvars / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvars.is_file():
        raise FileNotFoundError(
            f"Test compiler environment script is missing: {vcvars}"
        )
    arguments = architecture
    if architecture == "arm64":
        version = prepared.get("VLLM_WINDOWS_MSVC_VERSION", "14.51.36231")
        sdk = prepared.get("VLLM_WINDOWS_SDK_VERSION", "10.0.26100.0")
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise ValueError("Invalid VLLM_WINDOWS_MSVC_VERSION for test compiler.")
        if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", sdk):
            raise ValueError("Invalid VLLM_WINDOWS_SDK_VERSION for test compiler.")
        arguments += f" {sdk} -vcvars_ver={version}"
    bootstrap = {
        **prepared,
        "VLLM_TEST_VCVARS": str(vcvars),
        "VLLM_TEST_VCVARS_ARGS": arguments,
    }
    if prepared.get("PROGRAMFILES(X86)"):
        installer = (
            Path(prepared["PROGRAMFILES(X86)"])
            / "Microsoft Visual Studio"
            / "Installer"
        )
        if (installer / "vswhere.exe").is_file():
            bootstrap["PATH"] = str(installer) + os.pathsep + prepared.get("PATH", "")
    result = subprocess.run(
        [
            required(prepared, "COMSPEC"),
            "/d",
            "/c",
            "call",
            str(HERE / "test-environment.cmd"),
        ],
        env=bootstrap,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Test compiler setup failed (exit {result.returncode}):\n{result.stdout}"
        )
    lines = result.stdout.splitlines()
    marker = "__VLLM_TEST_ENVIRONMENT__"
    if marker not in lines:
        raise RuntimeError("Test compiler setup did not return its environment.")
    for line in lines[lines.index(marker) + 1 :]:
        name, separator, value = line.partition("=")
        if name and separator and not name.startswith("VLLM_TEST_"):
            prepared[name.upper()] = value
    compiler = shutil.which("cl.exe", path=prepared.get("PATH", ""))
    if (
        prepared.get("VSCMD_ARG_TGT_ARCH", "").lower() != architecture
        or not prepared.get("INCLUDE")
        or not prepared.get("LIB")
        or not compiler
    ):
        raise RuntimeError(
            f"Visual Studio did not initialize the {architecture} compiler, headers and libraries."
        )
    prepared["CC"] = compiler
    prepared["CXX"] = compiler
    print(f"Pytest JIT compiler: {compiler}", flush=True)
    return prepared


def installed_test_packages(python, env):
    return json.loads(
        run(
            [
                python,
                "-I",
                "-c",
                (
                    "import importlib.metadata as m, json, re; "
                    "names = {d.metadata['Name'] for d in m.distributions()}; "
                    "print(json.dumps({re.sub(r'[-_.]+', '-', n).lower(): m.version(n) "
                    "for n in names}, sort_keys=True))"
                ),
            ],
            env=env,
            capture=True,
        )
    )


def run_full_tests(
    python, venv, source, wheel, work, output, env, architecture, test_suite="full"
):
    summary_path = output / "pytest-summary.json"
    summary = {"suite": test_suite, "status": "preparing-test-environment"}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    test_wheels = [wheel]
    if architecture == "arm64":
        torchvision = (
            HERE
            / "wheelhouse"
            / "torchvision-0.29.0a0+10f68db-cp313-cp313-win_arm64.whl"
        )
        if not torchvision.is_file():
            raise FileNotFoundError(
                f"Bundled ARM64 TorchVision wheel is missing: {torchvision}"
            )
        test_wheels.append(torchvision)
    test_utilities = source / "tests" / "vllm_test_utils"
    if not (test_utilities / "setup.py").is_file():
        raise FileNotFoundError(
            f"The selected source has no test utility package: {test_utilities}"
        )
    compiler_env = test_compiler_environment(architecture, env)
    test_venv = Path(work) / "test-venv"
    run([python, "-I", "-m", "venv", test_venv], cwd=work, env=env)
    test_python = test_venv / "Scripts" / "python.exe"
    test_site = test_venv / "Lib" / "site-packages"
    # Nested venv --system-site-packages would inherit the system Python, not
    # the pool venv. Share only the pool's preinstalled dependencies, read-only.
    (test_site / "pool-dependencies.pth").write_text(
        str(venv / "Lib" / "site-packages") + "\n", encoding="utf-8"
    )
    test_env = dict(compiler_env)
    test_env["VIRTUAL_ENV"] = str(test_venv)
    test_env["PATH"] = str(test_python.parent) + os.pathsep + compiler_env["PATH"]
    test_env.pop("PYTHONPATH", None)
    test_env.pop("PYTEST_ADDOPTS", None)
    # Make the supplied wheels visible before resolving their test dependencies.
    run(
        [
            test_python,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--only-binary=:all:",
            "--no-deps",
            "--no-compile",
            "--ignore-installed",
            *test_wheels,
        ],
        cwd=work,
        env=test_env,
    )
    # This is the pinned checkout's pure-Python helper, not a native dependency.
    run(
        [
            test_python,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--no-compile",
            "--ignore-installed",
            test_utilities,
        ],
        cwd=work,
        env=test_env,
    )
    runtime_packages = installed_test_packages(test_python, test_env)
    if "torch" not in runtime_packages:
        raise RuntimeError(
            "The test venv cannot see the provisioned Torch distribution."
        )
    constraints = output / "test-runtime-constraints.txt"
    constraints.write_text(
        "".join(
            f"{name}=={version}\n" for name, version in sorted(runtime_packages.items())
        ),
        encoding="utf-8",
    )
    install = [
        test_python,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--only-binary=:all:",
        "-c",
        constraints,
        "-r",
        HERE / "requirements-tests.txt",
    ]
    if env.get("VLLM_WINDOWS_WHEELHOUSE"):
        install += ["--find-links", env["VLLM_WINDOWS_WHEELHOUSE"]]
    if env.get("VLLM_WINDOWS_NO_INDEX") == "1":
        install += ["--no-index"]
    run(install, cwd=work, env=test_env)
    installed_packages = installed_test_packages(test_python, test_env)
    (output / "test-dependencies.json").write_text(
        json.dumps(
            {"protected_runtime": runtime_packages, "installed": installed_packages},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    changed = {
        name: {"expected": version, "installed": installed_packages.get(name)}
        for name, version in runtime_packages.items()
        if installed_packages.get(name) != version
    }
    if changed:
        raise RuntimeError(
            f"Test installation changed protected runtime packages: {changed}"
        )
    shutil.copyfile(HERE / "wheel_imports.py", test_site / "_vllm_ci_wheel_imports.py")
    shutil.copyfile(
        HERE / "windows_helpers.py", test_site / "_vllm_ci_windows_helpers.py"
    )
    (test_site / "vllm-ci-wheel.pth").write_text(
        "import _vllm_ci_wheel_imports; import _vllm_ci_windows_helpers\n",
        encoding="utf-8",
    )
    run(
        [
            test_python,
            "-I",
            HERE / "run_tests.py",
            "--source-dir",
            source,
            "--output",
            output,
            "--test-suite",
            test_suite,
        ],
        cwd=source,
        env=test_env,
    )


def build(architecture, test_suite="smoke"):
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
    runtime = probe_runtime(python, env, architecture, require_gpu=test_suite != "none")
    (output / "runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    run([Path(env["CUDA_PATH"]) / "bin" / "nvcc.exe", "--version"], env=env)

    work_root = Path(required(env, "VLLM_WINDOWS_WORK_ROOT")).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vllm-", dir=work_root) as work:
        env = git_longpaths_environment(env, Path(work))
        source = Path(work) / "src"
        checkout_source(source, source_info, env)
        command = build_command(architecture, source, python, venv, output, env)
        run(command, cwd=source, env=env)
        wheel = validate_wheel(output, architecture)
        if test_suite == "none":
            publish_build_manifest(
                output, architecture, wheel, source_info, runtime, env
            )
            return
        if test_suite == "full":
            run_full_tests(python, venv, source, wheel, work, output, env, architecture)
            return
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
    build_parser.add_argument(
        "--test-suite", choices=["none", "smoke", "full"], default="smoke"
    )
    test_parser = subcommands.add_parser("test")
    test_parser.add_argument("--architecture", choices=PLATFORMS, required=True)
    test_parser.add_argument(
        "--test-suite",
        choices=["full", "deepgemm"],
        default="full",
        help="Select the full tree or only the DeepGEMM SiLU/FP8 reference-scale tests.",
    )
    test_parser.add_argument(
        "--source-build-id",
        help=f"Reuse a previous Buildkite build's artifacts (also via {TEST_BUILD_ID_ENV}).",
    )
    args = parser.parse_args()
    if args.command == "resolve":
        resolve_source()
    elif args.command == "test":
        test_build(args.architecture, args.source_build_id, args.test_suite)
    else:
        build(args.architecture, args.test_suite)


if __name__ == "__main__":
    main()
