import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

WINDOWS = Path(__file__).resolve().parents[1] / "windows"
PROVISION = WINDOWS / "provision-pool.ps1"
LAUNCHER = WINDOWS / "start-agent.ps1"
TOOLCHAINS = WINDOWS / "install-toolchains.ps1"
FIND_TOOLS = WINDOWS / "find-toolchains.ps1"
PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(not PWSH, reason="Requires PowerShell 7")
TORCH_INDEX = "https://pypi.nvidia.cn/nvtorch_oot_nightly/"
TORCH_VERSION = "2.16.0.dev20261009+cu134"
TORCH_URL = (
    TORCH_INDEX + "torch/torch-2.16.0.dev20261009%2Bcu134-cp313-cp313-win_arm64.whl"
)
TORCH_HASH = "a" * 64


def torch_report():
    return {
        "install": [
            {
                "metadata": {"name": "torch", "version": TORCH_VERSION},
                "download_info": {
                    "url": TORCH_URL,
                    "archive_info": {"hashes": {"sha256": TORCH_HASH}},
                },
            }
        ]
    }


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(tmp_path, script):
    harness = tmp_path / "harness.ps1"
    harness.write_text("$ErrorActionPreference = 'Stop'\n" + script, encoding="utf-8")
    return subprocess.run(
        [PWSH, "-NoLogo", "-NoProfile", "-NonInteractive", "-File", str(harness)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def load_provision(tmp_path, architecture="x64"):
    return (
        f". {literal(PROVISION)} -Architecture {architecture} "
        "-PythonExecutable python.exe -CudaPath unused "
        "-VisualStudioPath unused -RequirementsFile unused "
        f"-CudaArchList '12.0+PTX;10.3a' -InstallRoot {literal(tmp_path / 'pool')}\n"
    )


def test_arm64_runtime_manifest_parses_and_does_not_pin_torch():
    from packaging.requirements import Requirement

    manifest = WINDOWS / "requirements-arm64.txt"
    requirements = [
        Requirement(line)
        for raw in manifest.read_text().splitlines()
        if (line := raw.split("#", 1)[0].strip())
    ]
    names = {req.name.lower().replace("_", "-") for req in requirements}
    assert len(names) == len(requirements)
    assert {"torch", "torchvision", "torchaudio", "vllm"}.isdisjoint(names)
    assert {
        "numpy",
        "psutil",
        "regex",
        "requests",
        "typing-extensions",
        "transformers",
        "tokenizers",
        "safetensors",
        "fastapi",
        "pydantic",
        "msgspec",
        "pyzmq",
        "triton-windows",
        "winloop",
        "portalocker",
    } <= names
    assert {"flashinfer-python", "xformers", "tilelang", "instanttensor"}.isdisjoint(
        names
    )
    assert {"opencv-python", "opencv-python-headless"}.isdisjoint(names)
    mistral = next(req for req in requirements if req.name == "mistral-common")
    assert not mistral.extras
    triton = next(req for req in requirements if req.name == "triton-windows")
    assert triton.specifier.contains("3.8.0.post29")


@pytest.mark.parametrize("custom", [False, True])
def test_arm64_runtime_requirements_default_or_override(tmp_path, custom):
    custom_file = tmp_path / "custom-requirements.txt"
    custom_file.write_text("requests\n")
    expected = custom_file if custom else WINDOWS / "requirements-arm64.txt"
    result = run_ps(
        tmp_path,
        load_provision(tmp_path, "arm64")
        + f"""
$RequirementsFile = {literal(custom_file) if custom else "''"}
function Assert-NativeHost {{}}
function Resolve-RequiredPath {{
    param($Path, $Type)
    if ($Path -ne {literal(expected)}) {{ throw "Wrong runtime manifest: $Path" }}
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {{ throw 'Missing default manifest' }}
    throw 'Reached expected runtime manifest before installing anything'
}}
Invoke-PoolProvisioning
""",
    )
    assert result.returncode != 0
    assert "Reached expected runtime manifest" in result.stderr
    assert not (tmp_path / "pool").exists()


@pytest.mark.parametrize("script", [PROVISION, LAUNCHER, TOOLCHAINS, FIND_TOOLS])
def test_powershell_syntax(tmp_path, script):
    result = run_ps(
        tmp_path,
        f"""
$tokens = $null
$errors = $null
[void][Management.Automation.Language.Parser]::ParseFile(
    {literal(script)}, [ref]$tokens, [ref]$errors)
if ($errors.Count) {{ throw ($errors | Out-String) }}
""",
    )
    assert result.returncode == 0, result.stderr


def test_provisioning_rejects_wrong_native_host(tmp_path):
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + """
$target = if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() -eq 'Arm64') {
    'x64'
} else { 'arm64' }
Assert-NativeHost $target
""",
    )
    assert result.returncode != 0
    assert "native PowerShell 7" in result.stderr
    assert not (tmp_path / "pool").exists()


def test_provisioning_never_overwrites_existing_directory(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    sentinel = pool / "user-owned.txt"
    sentinel.write_text("preserve")
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + """
function Assert-NativeHost {}
Invoke-PoolProvisioning
""",
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert sentinel.read_text() == "preserve"
    assert list(pool.iterdir()) == [sentinel]


def test_checked_command_propagates_native_failure(tmp_path):
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + f"Invoke-CheckedCommand {literal(sys.executable)} @('-c', 'import sys; sys.exit(17)')\n",
    )
    assert result.returncode != 0
    assert "exit code 17" in result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises Windows directory ACLs")
def test_pool_directory_permissions_are_private(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + """
Protect-PoolDirectory $InstallRoot
$acl = Get-Acl -LiteralPath $InstallRoot
if (-not $acl.AreAccessRulesProtected) { throw 'Inherited permissions still enabled' }
$allowed = @(
    [Security.Principal.WindowsIdentity]::GetCurrent().User.Value,
    'S-1-5-18', 'S-1-5-32-544'
)
foreach ($rule in $acl.Access) {
    $sid = $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
    if ($sid -notin $allowed -or $rule.IsInherited) { throw "Unexpected permission: $sid" }
}
'still writable' | Set-Content -LiteralPath (Join-Path $InstallRoot 'writable.txt')
""",
    )
    assert result.returncode == 0, result.stderr
    assert (pool / "writable.txt").read_text().strip() == "still writable"


@pytest.mark.parametrize("missing_compiler", [True, False])
def test_arm64_toolset_checks_target_compiler(tmp_path, missing_compiler):
    vs = tmp_path / "Visual Studio"
    vcvars = vs / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    vcvars.parent.mkdir(parents=True)
    vcvars.touch()
    toolset = vs / "VC" / "Tools" / "MSVC" / "14.51.36231"
    (toolset / "lib" / "arm64").mkdir(parents=True)
    if not missing_compiler:
        compiler = toolset / "bin" / "HostARM64" / "arm64" / "cl.exe"
        compiler.parent.mkdir(parents=True)
        compiler.touch()
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + f"Assert-Arm64Toolset {literal(vcvars)} '14.51.36231'\n",
    )
    if missing_compiler:
        assert result.returncode != 0
        assert "ARM64 MSVC compiler" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("parameter", ["VisualStudioPath", "RustVisualStudioPath"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("use_file", [False, True])
def test_resolve_vcvars_reports_incomplete_installation(
    tmp_path, parameter, missing, use_file
):
    vs = tmp_path / "Visual Studio"
    vcvars = vs / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    vcvars.parent.mkdir(parents=True)
    if not missing:
        vcvars.touch()
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + f"""
$path = Resolve-VcVars {literal(vcvars if use_file else vs)} -ParameterName '{parameter}'
if ($path -ne {literal(vcvars)}) {{ throw 'Incorrect environment script' }}
""",
    )
    if missing:
        assert result.returncode != 0
        assert parameter in result.stderr
        assert "Rerun -InstallToolchains as Administrator" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("architecture", ["x64", "arm64"])
@pytest.mark.parametrize("bad_checksum", [False, True])
def test_agent_download_verifies_checksum_before_extracting(
    tmp_path, architecture, bad_checksum
):
    archive = tmp_path / "release.zip"
    payload = b"fake agent, never executed"
    with zipfile.ZipFile(archive, "w") as wheel:
        wheel.writestr("buildkite-agent.exe", payload)
        wheel.writestr("ignored.cfg", "do not install upstream configuration")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    release_arch = "amd64" if architecture == "x64" else "arm64"
    expected_name = f"buildkite-agent-windows-{release_arch}-4.3.0.zip"
    (tmp_path / "pool").mkdir()
    result = run_ps(
        tmp_path,
        load_provision(tmp_path, architecture)
        + f"""
function Invoke-WebRequest {{
    param($Uri, $OutFile)
    if ($Uri -like '*.SHA256SUMS') {{
        {literal(("0" * 64 if bad_checksum else digest) + "  " + expected_name)} |
            Set-Content -LiteralPath $OutFile
    }} else {{
        if ($Uri -ne 'https://github.com/buildkite/agent/releases/download/v4.3.0/{expected_name}') {{
            throw 'Wrong release architecture or URL'
        }}
        Copy-Item -LiteralPath {literal(archive)} -Destination $OutFile
    }}
}}
function Invoke-CheckedCommand {{
    param($File, $Arguments)
    if ($Arguments[0] -ne '--version') {{ throw 'Agent must not be started' }}
}}
$hash = Install-PoolAgent $InstallRoot $Architecture $AgentVersion
if ($hash -ine {literal(digest)}) {{ throw 'Incorrect recorded hash' }}
""",
    )
    binary = tmp_path / "pool" / "bin" / "buildkite-agent.exe"
    if bad_checksum:
        assert result.returncode != 0
        assert "checksum verification failed" in result.stderr
        assert not binary.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert binary.read_bytes() == payload
        assert not (tmp_path / "pool" / "ignored.cfg").exists()
        assert not list((tmp_path / "pool").glob("*.zip"))


@pytest.mark.parametrize(
    "architecture,use_default,no_index",
    [
        ("x64", False, True),
        ("arm64", False, True),
        ("arm64", True, True),
        ("arm64", True, False),
    ],
)
@pytest.mark.parametrize(
    "failure", [None, "pip", "runtime", "not-venv", "wrong-venv-version"]
)
def test_provisioning_writes_complete_secret_free_pool(
    tmp_path, architecture, use_default, no_index, failure
):
    inputs = tmp_path / "inputs with spaces"
    inputs.mkdir()
    vcvars = inputs / "VS" / "vcvarsall.bat"
    vcvars.parent.mkdir()
    vcvars.touch()
    requirements = inputs / "requirements.txt"
    requirements.write_text("placeholder manifest; native installs are mocked")
    if use_default:
        requirements = WINDOWS / "requirements-arm64.txt"
    platform = "win-amd64" if architecture == "x64" else "win-arm64"
    rust_host = (
        "x86_64-pc-windows-msvc" if architecture == "x64" else "aarch64-pc-windows-msvc"
    )
    runtime = {
        "platform": platform,
        "cuda": "13.4" if failure != "runtime" else "12.0",
        "torch": "private-build",
        "gpu": "test GPU",
    }
    result = run_ps(
        tmp_path,
        load_provision(tmp_path, architecture)
        + f"""
$RequirementsFile = {literal(requirements) if not use_default else "''"}
$Wheelhouse = {literal(inputs)}
$NoIndex = ${str(no_index).lower()}
$CMakeCudaArchitectures = '120-real;103-real'
$CudaPath = {literal(inputs / "CUDA")}
$VisualStudioPath = {literal(vcvars)}
$ProtocIncludePath = {literal(inputs / "include")}
$env:BUILDKITE_AGENT_TOKEN = 'unit-test-token-not-for-disk'
$env:CUDA_HOME = 'original-cuda-home'
$env:CARGO_HOME = 'original-cargo-home'
$originalPath = $env:PATH
${{env:ProgramFiles(x86)}} = {literal(inputs)}
$script:commands = [Collections.Generic.List[object]]::new()
function Assert-NativeHost {{}}
function Assert-Arm64Toolset {{
    param($VcVars, $Version)
    if ($Version -notin @('14.51.36231', '14.44.35207')) {{ throw 'Wrong toolset' }}
}}
function Resolve-Tool {{ param($Path); return (Join-Path {literal(inputs)} $Path) }}
function Resolve-RequiredPath {{ param($Path, $Type); return $Path }}
function Protect-PoolDirectory {{}}
function Invoke-CheckedCommand {{
    param($File, $Arguments, [switch]$Capture)
    $script:commands.Add(@{{file=$File; arguments=@($Arguments)}})
    if ($File -like '*nvcc.exe') {{ return 'Cuda compilation tools, release 13.4' }}
    if ($File -like '*rustc.exe') {{ return "host: {rust_host}`nrelease: 1.95.0" }}
    if ($Arguments -contains 'venv') {{
        New-Item -ItemType Directory -Path (Join-Path $InstallRoot 'venv\\Scripts') | Out-Null
        return
    }}
    if ($Arguments -contains '--dry-run') {{
        {literal(json.dumps(torch_report()))} |
            Set-Content -LiteralPath $Arguments[[array]::IndexOf($Arguments, '--report') + 1]
        return
    }}
    if ($Arguments -contains 'install' -and {literal(failure or "")} -eq 'pip') {{
        throw 'Simulated dependency installation failure'
    }}
    if ($Arguments -contains 'install') {{
        if (-not $env:PATH.Contains({literal(inputs)}) -or $env:CUDA_HOME -ne $CudaPath) {{
            throw 'New toolchains must be available during dependency installation'
        }}
        if ($env:CARGO_HOME -ne (Join-Path $InstallRoot 'cargo-cache')) {{
            throw 'Dependency builds must use a writable per-agent Cargo cache'
        }}
    }}
    if ($Arguments -contains '-c') {{
        if ($Arguments[-1] -like '*sys.version_info*') {{
            $info = @{{platform='{platform}'; version=@(3,13,16); is_venv=($File -like '*venv*')}}
            if ($info.is_venv) {{
                if ('{failure}' -eq 'not-venv') {{ $info.is_venv = $false }}
                if ('{failure}' -eq 'wrong-venv-version') {{ $info.version = @(3,12,10) }}
            }}
            return ($info | ConvertTo-Json)
        }}
        return {literal(json.dumps(runtime))}
    }}
}}
function Install-PoolAgent {{
    param($Root, $Target, $Version)
    New-Item -ItemType Directory -Path (Join-Path $Root 'bin') | Out-Null
    New-Item -ItemType File -Path (Join-Path $Root 'bin\\buildkite-agent.exe') | Out-Null
    return ('a' * 64)
}}
try {{
    Invoke-PoolProvisioning
}} finally {{
    if ($env:PATH -ne $originalPath -or $env:CUDA_HOME -ne 'original-cuda-home' -or
        $env:CARGO_HOME -ne 'original-cargo-home') {{ throw 'Leaked dependency environment' }}
}}
$script:commands | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath {literal(tmp_path / "commands.json")}
""",
    )
    pool = tmp_path / "pool"
    if failure:
        assert result.returncode != 0
        if failure in ("not-venv", "wrong-venv-version"):
            assert "The new venv must use the selected native Python" in result.stderr
        assert not (pool / "provisioning.json").exists()
        assert not (pool / "start-agent.ps1").exists()
        return
    assert result.returncode == 0, result.stderr
    assert f"Using build/runtime wheelhouse: {inputs}" in result.stdout
    settings = json.loads((pool / "environment.json").read_text(encoding="utf-8"))
    env = settings["environment"]
    assert Path(env["VLLM_WINDOWS_VENV"]) == pool / "venv"
    assert Path(env["VLLM_WINDOWS_WORK_ROOT"]) == pool / "work"
    assert Path(env["BUILDKITE_BUILD_PATH"]) == pool / "checkouts"
    assert "BUILDKITE_AGENT_TOKEN" not in env
    assert "PATH" not in env
    assert str(pool / "venv" / "Scripts") in settings["paths"]
    assert str(pool / "bin") in settings["paths"]
    if architecture == "arm64":
        assert env["VLLM_WINDOWS_MSVC_VERSION"] == "14.51.36231"
        assert env["VLLM_WINDOWS_RUST_MSVC_VERSION"] == "14.44.35207"
        assert env["CMAKE_CUDA_ARCHITECTURES"] == "120-real;103-real"
        assert "VLLM_WINDOWS_VCVARSALL" not in env
    else:
        assert env["VLLM_WINDOWS_VCVARSALL"].endswith("vcvarsall.bat")
        assert "VLLM_WINDOWS_MSVC_VERSION" not in env
    config = (pool / "buildkite-agent.cfg").read_text(encoding="utf-8")
    assert f'queue="windows-{architecture}"' in config
    assert 'git-clean-flags="-ffdx"' in config
    assert "pwsh.exe" in config
    assert (pool / "start-agent.ps1").read_bytes() == LAUNCHER.read_bytes()
    metadata = json.loads((pool / "provisioning.json").read_text(encoding="utf-8"))
    assert metadata["runtime"] == runtime
    if architecture == "arm64":
        assert metadata["torch_selection"]["Version"] == TORCH_VERSION
        assert (pool / "torch-constraints.txt").read_text().strip() == (
            f"torch @ {TORCH_URL}#sha256={TORCH_HASH}"
        )
    else:
        assert metadata["torch_selection"] is None
        assert not (pool / "torch-selection.json").exists()
        assert not (pool / "torch-constraints.txt").exists()
    assert (
        metadata["requirements_sha256"].lower()
        == hashlib.sha256(requirements.read_bytes()).hexdigest()
    )
    for file in pool.iterdir():
        if file.is_file():
            assert "unit-test-token-not-for-disk" not in file.read_text(
                encoding="utf-8"
            )
    commands = json.loads((tmp_path / "commands.json").read_text(encoding="utf-8"))
    creation = next(c for c in commands if "venv" in c["arguments"])
    assert creation["arguments"] == ["-I", "-m", "venv", str(pool / "venv")]
    assert "--system-site-packages" not in creation["arguments"]
    for command in commands:
        if "pip" in command["arguments"]:
            assert Path(command["file"]) == pool / "venv" / "Scripts" / "python.exe"
    installation = next(
        c["arguments"]
        for c in commands
        if "install" in c["arguments"] and "--dry-run" not in c["arguments"]
    )
    assert "--isolated" in installation
    assert str(WINDOWS / "requirements-toolchain.txt") in installation
    assert str(requirements) in installation
    assert ("--no-index" in installation) == no_index
    assert installation[installation.index("--find-links") + 1] == str(inputs)
    selection = [c["arguments"] for c in commands if "--dry-run" in c["arguments"]]
    if architecture == "arm64":
        assert len(selection) == 1
        assert selection[0][selection[0].index("--index-url") + 1] == TORCH_INDEX
        assert "--no-index" not in selection[0]
        assert "--find-links" not in selection[0]
        assert "--pre" in selection[0]
        assert "--ignore-installed" in selection[0]
        assert "--only-binary=:all:" in selection[0]
        assert "--no-deps" in selection[0]
        assert selection[0][-1] == "torch"
        assert f"torch=={TORCH_VERSION}" in installation
        assert installation[installation.index("-c") + 1] == str(
            pool / "torch-constraints.txt"
        )
    else:
        assert not selection
        assert "-c" not in installation
    assert any("check" in c["arguments"] for c in commands)


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "wrong-origin",
        "x64-wheel",
        "missing-hash",
        "wrong-package",
        "invalid-version",
    ],
)
def test_nvidia_torch_selection_pins_exact_source_and_hash(tmp_path, invalid):
    pool = tmp_path / "pool"
    pool.mkdir()
    report = torch_report()
    package = report["install"][0]
    if invalid == "wrong-origin":
        package["download_info"]["url"] = TORCH_URL.replace(
            "pypi.nvidia.cn", "example.org"
        )
    elif invalid == "x64-wheel":
        package["download_info"]["url"] = TORCH_URL.replace("win_arm64", "win_amd64")
    elif invalid == "missing-hash":
        package["download_info"]["archive_info"]["hashes"]["sha256"] = ""
    elif invalid == "wrong-package":
        package["metadata"]["name"] = "torchvision"
    elif invalid == "invalid-version":
        package["metadata"]["version"] = "2.16\nother-package"
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + f"""
function Invoke-CheckedCommand {{
    param($File, $Arguments)
    if ($File -ne 'venv-python.exe' -or $Arguments[-1] -ne 'torch') {{ throw 'Wrong selection command' }}
    if ($Arguments -contains '-r' -or $Arguments -contains '-c' -or $Arguments -contains '--extra-index-url') {{
        throw 'Do not let runtime pins or other indexes affect the latest Torch selection'
    }}
    {literal(json.dumps(report))} |
        Set-Content -LiteralPath $Arguments[[array]::IndexOf($Arguments, '--report') + 1]
}}
$selection = Select-NvidiaTorch 'venv-python.exe' {literal(pool)}
if ($selection.Version -ne '{TORCH_VERSION}') {{ throw 'Wrong selected version' }}
""",
    )
    if invalid:
        assert result.returncode != 0
        assert not (pool / "torch-constraints.txt").exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (pool / "torch-constraints.txt").read_text().strip() == (
            f"torch @ {TORCH_URL}#sha256={TORCH_HASH}"
        )


def test_nvidia_torch_selection_failure_has_no_fallback(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + """
$script:calls = 0
function Invoke-CheckedCommand {
    $script:calls++
    throw 'No compatible NVIDIA wheel available'
}
try {
    Select-NvidiaTorch 'venv-python.exe' $InstallRoot
} finally {
    if ($script:calls -ne 1) { throw 'Unexpected fallback installation' }
}
""",
    )
    assert result.returncode != 0
    assert "No compatible NVIDIA wheel available" in result.stderr
    assert not (pool / "torch-constraints.txt").exists()


@pytest.mark.parametrize("conflicting_pin", [False, True])
def test_pip_resolves_nightly_torch_url_constraint_without_downgrade(
    tmp_path, conflicting_pin
):
    wheel = tmp_path / f"torch-{TORCH_VERSION}-py3-none-any.whl"
    metadata_dir = f"torch-{TORCH_VERSION}.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{metadata_dir}/METADATA",
            f"Metadata-Version: 2.1\nName: torch\nVersion: {TORCH_VERSION}\n",
        )
        archive.writestr(
            f"{metadata_dir}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{metadata_dir}/RECORD", "")
    checksum = hashlib.sha256(wheel.read_bytes()).hexdigest()
    constraints = tmp_path / "torch-constraints.txt"
    constraints.write_text(f"torch @ {wheel.as_uri()}#sha256={checksum}\n")
    requirements = tmp_path / "runtime.txt"
    requirements.write_text("torch==2.11.0\n" if conflicting_pin else "torch\n")
    report = tmp_path / "resolution.json"
    # Exercise pip's direct-URL constraint behavior without installing a package
    # or accessing an external index.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--no-index",
            "--no-deps",
            "-r",
            str(requirements),
            "-c",
            str(constraints),
            f"torch=={TORCH_VERSION}",
            "--report",
            str(report),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if conflicting_pin:
        assert result.returncode != 0
        assert "ResolutionImpossible" in result.stderr
        assert not report.exists()
    else:
        assert result.returncode == 0, result.stderr
        selected = json.loads(report.read_text())["install"]
        assert len(selected) == 1
        assert selected[0]["metadata"]["version"] == TORCH_VERSION
        assert (
            selected[0]["download_info"]["archive_info"]["hashes"]["sha256"] == checksum
        )


@pytest.mark.parametrize("version", [[3, 12, 10], [3, 14, 8], [3, 13, 15]])
def test_pool_rejects_old_or_wrong_python_before_creating_venv(tmp_path, version):
    result = run_ps(
        tmp_path,
        load_provision(tmp_path)
        + f"""
function Assert-NativeHost {{}}
function Resolve-Tool {{ return 'test-python.exe' }}
function Resolve-RequiredPath {{ param($Path, $Type); return $Path }}
function Invoke-CheckedCommand {{
    param($File, $Arguments)
    if ($Arguments -contains 'venv' -or $Arguments -contains 'pip') {{
        throw 'Must reject the Python version before installing anything'
    }}
    return {literal(json.dumps({"platform": "win-amd64", "version": version, "is_venv": False}))}
}}
Invoke-PoolProvisioning
""",
    )
    assert result.returncode != 0
    assert "PythonExecutable must be native win-amd64 Python 3.13" in result.stderr
    assert not (tmp_path / "pool").exists()


@pytest.mark.parametrize("failure", [None, "missing-token", "missing-marker"])
def test_launcher_preserves_exit_status_and_restores_environment(tmp_path, failure):
    pool = tmp_path / "pool"
    (pool / "bin").mkdir(parents=True)
    for name in ["provisioning.json", "buildkite-agent.cfg", "bin/buildkite-agent.exe"]:
        (pool / name).write_text("")
    if failure == "missing-marker":
        (pool / "provisioning.json").unlink()
    (pool / "environment.json").write_text(
        json.dumps(
            {
                "environment": {
                    "VLLM_WINDOWS_VENV": str(pool / "venv"),
                    "VLLM_WINDOWS_WORK_ROOT": str(pool / "work"),
                },
                "paths": [str(pool / "bin"), str(pool / "venv" / "Scripts")],
            }
        )
    )
    result = run_ps(
        tmp_path,
        f"""
. {literal(LAUNCHER)}
$env:BUILDKITE_AGENT_TOKEN = {literal("" if failure == "missing-token" else "test-runtime-token")}
$env:VLLM_WINDOWS_VENV = 'original-venv'
[Environment]::SetEnvironmentVariable('VLLM_WINDOWS_WORK_ROOT', $null, 'Process')
$originalPath = $env:PATH
function Invoke-PoolAgent {{
    param($Executable, $Config)
    if ($env:VLLM_WINDOWS_VENV -ne {literal(str(pool / "venv"))}) {{
        throw 'Agent environment was not loaded'
    }}
    if (-not $env:PATH.StartsWith({literal(str(pool / "bin"))})) {{ throw 'Wrong PATH' }}
    if ($Executable -ne {literal(pool / "bin" / "buildkite-agent.exe")}) {{ throw 'Wrong executable' }}
    if ($Config -ne {literal(pool / "buildkite-agent.cfg")}) {{ throw 'Wrong configuration' }}
    return 23
}}
$result = Start-PoolAgent {literal(pool)}
if ($result -ne 23) {{ throw 'Lost agent exit status' }}
if ($env:VLLM_WINDOWS_VENV -ne 'original-venv') {{ throw 'Leaked modified environment' }}
if ($env:VLLM_WINDOWS_WORK_ROOT) {{ throw 'Leaked new environment variable' }}
if ($env:PATH -ne $originalPath) {{ throw 'Leaked PATH' }}
""",
    )
    if failure:
        assert result.returncode != 0
        expected = (
            "Supply BUILDKITE_AGENT_TOKEN"
            if failure == "missing-token"
            else "incomplete"
        )
        assert expected in result.stderr
    else:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("valid", [True, False])
def test_toolchain_payload_checksum(tmp_path, valid):
    payload = tmp_path / "payload.zip"
    payload.write_bytes(b"payload")
    checksum = hashlib.sha256(payload.read_bytes()).hexdigest() if valid else "0" * 64
    result = run_ps(
        tmp_path,
        f". {literal(TOOLCHAINS)}\nAssert-Payload {literal(payload)} {literal(checksum)} ''\n",
    )
    assert (result.returncode == 0) == valid, result.stderr


@pytest.mark.parametrize(
    "status,publisher,success",
    [
        ("Valid", "Microsoft Corporation", True),
        ("NotSigned", "Microsoft Corporation", False),
        ("Valid", "Unrelated Publisher", False),
    ],
)
def test_toolchain_payload_requires_expected_signature(
    tmp_path, status, publisher, success
):
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
function Get-AuthenticodeSignature {{
    return [pscustomobject]@{{
        Status = {literal(status)}
        SignerCertificate = [pscustomobject]@{{Subject = {literal("CN=" + publisher)}}}
    }}
}}
Assert-Payload 'unexecuted.exe' '' 'Microsoft Corporation'
""",
    )
    assert (result.returncode == 0) == success, result.stderr


@pytest.mark.parametrize("exit_code", [0, 3010, 1603, 1641])
def test_toolchain_installer_handles_reboot_and_errors(tmp_path, exit_code):
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
$script:ToolchainRebootRequired = $false
function Start-Process {{
    param($FilePath, $ArgumentList, [switch]$Wait, [switch]$PassThru)
    if (-not $Wait -or -not $PassThru) {{ throw 'Installer must be awaited' }}
    return [pscustomobject]@{{ExitCode = {exit_code}}}
}}
Invoke-ToolchainInstaller 'unexecuted.exe' @('--quiet', '--norestart')
if ($script:ToolchainRebootRequired -ne ${str(exit_code == 3010).lower()}) {{
    throw 'Incorrect reboot status'
}}
""",
    )
    assert (result.returncode == 0) == (exit_code in (0, 3010)), result.stderr
    if exit_code not in (0, 3010):
        assert f"exit code {exit_code}" in result.stderr


@pytest.mark.skipif(
    sys.platform != "win32", reason="Exercises native Windows argument quoting"
)
def test_installer_keeps_spaces_quotes_and_trailing_backslash(tmp_path):
    recorder = tmp_path / "record arguments.py"
    recorder.write_text(
        "import json,sys,pathlib\n"
        f"pathlib.Path({str(tmp_path / 'args.json')!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    values = ["TargetDir=C:\\tools with spaces\\", 'embedded "quote"', "one&two"]
    result = run_ps(
        tmp_path,
        f". {literal(TOOLCHAINS)}\nInvoke-ToolchainInstaller {literal(sys.executable)} "
        + "@("
        + ", ".join(literal(value) for value in [str(recorder), *values])
        + ")\n",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "args.json").read_text()) == values


@pytest.mark.parametrize("architecture", ["x64", "arm64"])
def test_msvc_installer_selects_cuda_and_rust_compilers(tmp_path, architecture):
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
function Get-VisualStudioPaths {{ return @() }}
$script:sdkInstalled = $false
function Test-WindowsSdk {{ return $script:sdkInstalled }}
function Get-ToolchainPayload {{ param($Url, $Path, $Sha256, $Publisher)
    if ($Publisher -ne 'Microsoft Corporation') {{ throw 'Unverified VS installer' }}
    if ($Url -like '*vs/18/release/*') {{ throw 'Nonexistent VS 2026 alias' }}
    return $Path
}}
function Invoke-ToolchainInstaller {{
    param($File, $Arguments)
    $script:sdkInstalled = $true
    $Arguments | ConvertTo-Json | Set-Content {literal(tmp_path / "vs-arguments.json")}
}}
function Get-InstalledMsvcVersion {{
    param($VisualStudio, $Family, $Target)
    return "$Family.12345"
}}
$result = Install-MsvcToolchains {literal(tmp_path)} {literal(tmp_path)} '{architecture}'
$result | ConvertTo-Json | Set-Content {literal(tmp_path / "vs-result.json")}
""",
    )
    assert result.returncode == 0, result.stderr
    arguments = json.loads((tmp_path / "vs-arguments.json").read_text())
    assert (
        "--quiet" in arguments and "--wait" in arguments and "--norestart" in arguments
    )
    assert "Microsoft.VisualStudio.Component.Windows11SDK.26100" in arguments
    assert "Microsoft.VisualStudio.Component.VC.CoreBuildTools" in arguments
    if architecture == "arm64":
        assert "Microsoft.VisualStudio.Component.VC.Tools.ARM64" in arguments
        assert "Microsoft.VisualStudio.Component.VC.14.44.17.14.ARM64" in arguments
    else:
        assert "Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64" in arguments
        assert "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in arguments
    versions = json.loads((tmp_path / "vs-result.json").read_text())
    assert versions["MsvcToolsetVersion"].startswith(
        "14.51" if architecture == "arm64" else "14.44"
    )
    assert versions["RustMsvcToolsetVersion"].startswith("14.44")


@pytest.mark.parametrize("valid_header", [True, False])
def test_cuda_header_fix_preserves_original_and_rejects_unknown_headers(
    tmp_path, valid_header
):
    header = tmp_path / "include" / "cuda.h"
    header.parent.mkdir()
    content = (
        "typedef struct CUtensorMap_st {\n alignas(128) char a[128];\n _Alignas(128) char b[128];\n};"
        if valid_header
        else "unknown CUDA header"
    )
    header.write_text(content)
    backup = tmp_path / "cuda.h.original"
    result = run_ps(
        tmp_path,
        f". {literal(TOOLCHAINS)}\nRepair-Cuda13Header {literal(tmp_path)} {literal(backup)}\n",
    )
    if valid_header:
        assert result.returncode == 0, result.stderr
        assert backup.read_text() == content
        assert "#define TENSOR_MAP_ALIGN 64" in header.read_text()
        assert "alignas(128)" not in header.read_text()
    else:
        assert result.returncode != 0
        assert header.read_text() == content
        assert not backup.exists()


@pytest.mark.parametrize("architecture", ["x64", "arm64"])
def test_generated_toolchain_configuration_is_imported(tmp_path, architecture):
    parameters = {
        "PythonExecutable": "native-python.exe",
        "GitExecutable": "installed-git.exe",
        "PwshExecutable": "native-pwsh.exe",
        "CargoExecutable": "installed-cargo.exe",
        "RustcExecutable": "installed-rustc.exe",
        "PerlPath": "installed-perl.exe",
        "ProtocPath": "installed-protoc.exe",
        "ProtocIncludePath": "protobuf-include",
        "MsvcToolsetVersion": "14.51.36231",
    }
    config = tmp_path / "toolchains.json"
    config.write_text(
        json.dumps({"architecture": architecture, "parameters": parameters})
    )
    result = run_ps(
        tmp_path,
        load_provision(tmp_path, architecture)
        + f"Import-ToolchainConfig {literal(config)} '{architecture}'\n"
        + "\n".join(
            f"if (${name} -ne {literal(value)}) {{ throw 'Not imported: {name}' }}"
            for name, value in parameters.items()
        ),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "bad_config",
    [
        {"architecture": "arm64", "parameters": {}},
        {"architecture": "x64", "parameters": {"InstallRoot": "C:\\unrelated"}},
    ],
)
def test_toolchain_configuration_rejects_wrong_architecture_or_unknown_keys(
    tmp_path, bad_config
):
    config = tmp_path / "toolchains.json"
    config.write_text(json.dumps(bad_config))
    result = run_ps(
        tmp_path,
        load_provision(tmp_path) + f"Import-ToolchainConfig {literal(config)} 'x64'\n",
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "platform,version,found",
    [
        ("win-arm64", "3.12.10", True),
        ("win-amd64", "3.12.10", False),
        ("win-arm64", "3.12.9", False),
    ],
)
def test_existing_python_must_match_native_architecture_and_version(
    tmp_path, platform, version, found
):
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
function Test-Path {{ return $true }}
function Get-ItemProperty {{ return [pscustomobject]@{{ExecutablePath='registered-python.exe'}} }}
function Get-PeArchitecture {{ return 'arm64' }}
function Invoke-ToolProbe {{ return {literal(version + " " + platform)} }}
$python = Get-MachinePython 'arm64' '3.12.10'
if ([bool]$python -ne ${str(found).lower()}) {{ throw 'Wrong Python was reused' }}
""",
    )
    assert result.returncode == 0, result.stderr


def test_cuda_installer_parameters_are_removed(tmp_path):
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
$entry = Get-Command {literal(PROVISION)}
$installer = Get-Command Install-WindowsToolchains
foreach ($name in @('CudaInstallerPath', 'CudaInstallerSha256', 'CudaInstaller')) {{
    if ($entry.Parameters.ContainsKey($name) -or $installer.Parameters.ContainsKey($name)) {{
        throw "Obsolete installer parameter: $name"
    }}
}}
""",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Uses built-in Windows PowerShell")
def test_install_mode_dispatches_from_windows_powershell_51(tmp_path):
    shutil.copyfile(PROVISION, tmp_path / "provision-pool.ps1")
    (tmp_path / "install-toolchains.ps1").write_text(
        "function Install-WindowsToolchains {\n"
        "param($Target,$Root,$CudaPath,$PythonVersion,$ExistingTools)\n"
        "if ($Target -ne 'arm64' -or $PythonVersion -ne '3.13.16') { throw 'Wrong arguments' }\n"
        "'dispatched without running installers' | Set-Content (Join-Path $PSScriptRoot 'dispatched.txt')\n"
        "return 3010\n"
        "}\n"
    )
    powershell = shutil.which("powershell.exe")
    result = subprocess.run(
        [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(tmp_path / "provision-pool.ps1"),
            "-Architecture",
            "arm64",
            "-InstallToolchains",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3010, result.stderr
    assert (tmp_path / "dispatched.txt").exists()


@pytest.mark.skipif(
    sys.platform != "win32", reason="Uses local Windows toolchain paths"
)
@pytest.mark.parametrize("architecture", ["x64", "arm64"])
@pytest.mark.parametrize("missing_rust_environment", [False, True])
def test_full_toolchain_orchestration_without_real_installers(
    tmp_path, architecture, missing_rust_environment
):
    root = tmp_path / "machine tools"
    old_python = root / "python" / "python.exe"
    old_installer = root / "downloads" / "python.exe"
    for file in (old_python, old_installer):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"old Python 3.12 must not be overwritten")
    (root / ".vllm-toolchains.json").write_text(
        json.dumps({"architecture": architecture})
    )
    cuda = tmp_path / "CUDA"
    sdk = tmp_path / "program-files-x86"
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
$script:downloads = [Collections.Generic.List[object]]::new()
$script:installers = [Collections.Generic.List[object]]::new()
$env:ProgramFiles = {literal(tmp_path)}
${{env:ProgramFiles(x86)}} = {literal(sdk)}
$env:CARGO_HOME = 'original-cargo'
$env:RUSTUP_HOME = 'original-rustup'
function Touch-File {{
    param($Path)
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $Path))
    [IO.File]::WriteAllText($Path, '')
}}
function Assert-ToolchainInstallerHost {{}}
function Set-ToolchainDirectoryPermissions {{}}
function Get-MachinePython {{ return $null }}
function Find-Tool {{ return $null }}
function Test-ToolCandidate {{ return $true }}
function Find-CudaToolkit {{
    param($Target, $Paths)
    if (Test-Path {literal(cuda / "bin" / "nvcc.exe")}) {{ return {literal(cuda)} }}
    return $null
}}
function Find-RustToolchain {{
    param($Target, $Paths)
    $triple = if ($Target -eq 'arm64') {{ 'aarch64-pc-windows-msvc' }} else {{ 'x86_64-pc-windows-msvc' }}
    $bin = Join-Path {literal(root)} "rustup\\toolchains\\1.95.0-$triple\\bin"
    if (Test-Path (Join-Path $bin 'rustc.exe')) {{
        return @{{RustcExecutable=(Join-Path $bin 'rustc.exe'); CargoExecutable=(Join-Path $bin 'cargo.exe')}}
    }}
    return $null
}}
function New-ItemProperty {{}}
function Get-ToolchainPayload {{
    param($Url, $Path, $Sha256, $Publisher)
    if (-not $Sha256 -and -not $Publisher) {{ throw 'Unverified download' }}
    $script:downloads.Add(@{{url=$Url; sha256=$Sha256; publisher=$Publisher}})
    Touch-File $Path
    return $Path
}}
function Expand-Archive {{
    param($LiteralPath, $DestinationPath, [switch]$Force)
    switch ([IO.Path]::GetFileNameWithoutExtension($LiteralPath)) {{
        'pwsh' {{ Touch-File (Join-Path $DestinationPath 'pwsh.exe') }}
        'git' {{ Touch-File (Join-Path $DestinationPath 'cmd\\git.exe') }}
        'perl' {{ Touch-File (Join-Path $DestinationPath 'perl\\bin\\perl.exe') }}
        'protobuf' {{
            Touch-File (Join-Path $DestinationPath 'bin\\protoc.exe')
            Touch-File (Join-Path $DestinationPath 'include\\google\\protobuf\\struct.proto')
        }}
    }}
}}
function Install-MsvcToolchains {{
    param($Root, $Downloads, $Target)
    $vs = Join-Path $Root 'vs'
    $rustVs = if (${str(missing_rust_environment).lower()}) {{ Join-Path $Root 'incomplete-rust-vs' }} else {{ $vs }}
    Touch-File (Join-Path $vs 'VC\\Auxiliary\\Build\\vcvarsall.bat')
    Touch-File (Join-Path ${{env:ProgramFiles(x86)}} 'Windows Kits\\10\\Include\\10.0.26100.0\\um\\Windows.h')
    return @{{VisualStudioPath=$vs; RustVisualStudioPath=$rustVs; MsvcToolsetVersion='14.51.36231';
             RustMsvcToolsetVersion='14.44.35207'; WindowsSdkVersion='10.0.26100.0'}}
}}
function Invoke-ToolchainInstaller {{
    param($File, $Arguments)
    $script:installers.Add(@{{file=$File; arguments=@($Arguments)}})
    if ($File -like '*python-*.exe') {{
        $destination = ($Arguments | Where-Object {{ $_ -like 'TargetDir=*' }}).Substring(10)
        Touch-File (Join-Path $destination 'python.exe')
    }} elseif ($File -like '*rustup-init.exe') {{
        $triple = $Arguments[[array]::IndexOf($Arguments, '--default-host') + 1]
        Touch-File (Join-Path $env:RUSTUP_HOME "toolchains\\1.95.0-$triple\\bin\\cargo.exe")
        Touch-File (Join-Path $env:RUSTUP_HOME "toolchains\\1.95.0-$triple\\bin\\rustc.exe")
    }} else {{
        if (($Arguments -join ',') -ne '-s,-n') {{ throw 'CUDA must not auto-reboot' }}
        Touch-File {literal(cuda / "bin" / "nvcc.exe")}
        Touch-File {literal(cuda / "lib" / architecture / "cudart.lib")}
        Touch-File {literal(cuda / "include" / "cuda.h")}
        Set-Content {literal(cuda / "include" / "cuda.h")} 'typedef struct CUtensorMap_st {{ alignas(128) _Alignas(128) }};'
    }}
}}
function Invoke-CheckedCommand {{
    param($File, $Arguments, [switch]$Capture)
    if ($File -like '*nvcc.exe') {{
        return 'Cuda compilation tools, release {"13.4" if architecture == "arm64" else "13.0"}'
    }}
    if ($File -like '*python.exe') {{
        return '["3.13.16", "{"win-arm64" if architecture == "arm64" else "win-amd64"}"]'
    }}
}}
$status = Install-WindowsToolchains '{architecture}' {literal(root)} `
    {literal(cuda)}
if ($status -ne 0) {{ throw 'Unexpected installer status' }}
if ($env:CARGO_HOME -ne 'original-cargo' -or $env:RUSTUP_HOME -ne 'original-rustup') {{
    throw 'Rust installation leaked environment changes'
}}
@{{downloads=@($script:downloads); installers=@($script:installers)}} | ConvertTo-Json -Depth 6 |
    Set-Content {literal(tmp_path / "operations.json")}
""",
    )
    if missing_rust_environment:
        assert result.returncode != 0
        assert "Installer did not produce the required file" in result.stderr
        assert "incomplete-rust-vs" in result.stderr
        assert not (root / "toolchains.json").exists()
        return
    assert result.returncode == 0, result.stderr
    configuration = json.loads(
        (root / "toolchains.json").read_text(encoding="utf-8-sig")
    )
    assert configuration["architecture"] == architecture
    parameters = configuration["parameters"]
    assert Path(parameters["PythonExecutable"]) == root / "python-3.13" / "python.exe"
    for file in (old_python, old_installer):
        assert file.read_bytes() == b"old Python 3.12 must not be overwritten"
    for key in [
        "PythonExecutable",
        "PwshExecutable",
        "GitExecutable",
        "CargoExecutable",
        "RustcExecutable",
        "PerlPath",
        "ProtocPath",
    ]:
        assert Path(parameters[key]).is_file()
    assert (
        Path(parameters["ProtocIncludePath"]) / "google" / "protobuf" / "struct.proto"
    ).exists()
    operations = json.loads((tmp_path / "operations.json").read_text())
    urls = [item["url"] for item in operations["downloads"]]
    assert any("strawberry-perl-" in url for url in urls)
    assert any("protoc-33.0-win64.zip" in url for url in urls)
    assert any(f"PowerShell-7.4.13-win-{architecture}.zip" in url for url in urls)
    assert any(
        f"MinGit-2.51.0-{'arm64' if architecture == 'arm64' else '64-bit'}.zip" in url
        for url in urls
    )
    assert any(
        f"python-3.13.16-{'arm64' if architecture == 'arm64' else 'amd64'}.exe" in url
        for url in urls
    )
    assert any("rustup-init.exe" in url for url in urls)
    cuda_url = (
        "https://developer.download.nvidia.com/compute/cuda/13.4.2/local_installers/cuda_13.4.2_windows_arm64.exe"
        if architecture == "arm64"
        else "https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_windows.exe"
    )
    cuda_downloads = [
        item
        for item in operations["downloads"]
        if "developer.download.nvidia.com" in item["url"]
    ]
    assert len(cuda_downloads) == 1
    assert cuda_downloads[0]["url"] == cuda_url
    assert cuda_downloads[0]["publisher"] == "NVIDIA Corporation"
    cuda_installs = [
        item
        for item in operations["installers"]
        if Path(item["file"]).name == cuda_url.rsplit("/", 1)[-1]
    ]
    assert len(cuda_installs) == 1
    assert cuda_installs[0]["arguments"] == ["-s", "-n"]
    if architecture == "x64":
        assert (root / "cuda.h.original").exists()
