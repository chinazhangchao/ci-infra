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
PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(not PWSH, reason="Requires PowerShell 7")


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


@pytest.mark.parametrize("script", [PROVISION, LAUNCHER])
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


@pytest.mark.parametrize("architecture", ["x64", "arm64"])
@pytest.mark.parametrize("failure", [None, "pip", "runtime"])
def test_provisioning_writes_complete_secret_free_pool(tmp_path, architecture, failure):
    inputs = tmp_path / "inputs with spaces"
    inputs.mkdir()
    requirements = inputs / "requirements.txt"
    requirements.write_text("placeholder manifest; native installs are mocked")
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
$RequirementsFile = {literal(requirements)}
$Wheelhouse = {literal(inputs)}
$NoIndex = $true
$CMakeCudaArchitectures = '120-real;103-real'
$CudaPath = {literal(inputs / "CUDA")}
$VisualStudioPath = {literal(inputs / "VS" / "vcvarsall.bat")}
$ProtocIncludePath = {literal(inputs / "include")}
$env:BUILDKITE_AGENT_TOKEN = 'unit-test-token-not-for-disk'
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
    if ($Arguments -contains 'install' -and {literal(failure or "")} -eq 'pip') {{
        throw 'Simulated dependency installation failure'
    }}
    if ($Arguments -contains '-c') {{
        if ($File -like '*venv*') {{ return {literal(json.dumps(runtime))} }}
        return {literal(json.dumps({"platform": platform, "version": [3, 12]}))}
    }}
}}
function Install-PoolAgent {{
    param($Root, $Target, $Version)
    New-Item -ItemType Directory -Path (Join-Path $Root 'bin') | Out-Null
    New-Item -ItemType File -Path (Join-Path $Root 'bin\\buildkite-agent.exe') | Out-Null
    return ('a' * 64)
}}
Invoke-PoolProvisioning
$script:commands | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath {literal(tmp_path / "commands.json")}
""",
    )
    pool = tmp_path / "pool"
    if failure:
        assert result.returncode != 0
        assert not (pool / "provisioning.json").exists()
        assert not (pool / "start-agent.ps1").exists()
        return
    assert result.returncode == 0, result.stderr
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
    installation = next(c["arguments"] for c in commands if "install" in c["arguments"])
    assert "--isolated" in installation
    assert "--no-index" in installation
    assert installation[installation.index("--find-links") + 1] == str(inputs)
    assert any("check" in c["arguments"] for c in commands)


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
