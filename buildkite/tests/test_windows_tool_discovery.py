import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

from buildkite.tests.test_windows_provisioning import (
    FIND_TOOLS,
    PWSH,
    TOOLCHAINS,
    literal,
    run_ps,
)

pytestmark = pytest.mark.skipif(not PWSH, reason="Requires PowerShell 7")


def pe_file(path, architecture):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(128)
    struct.pack_into("<H", header, 0, 0x5A4D)
    struct.pack_into("<I", header, 60, 80)
    struct.pack_into(
        "<IH", header, 80, 0x4550, {"x64": 0x8664, "arm64": 0xAA64}[architecture]
    )
    path.write_bytes(header)


@pytest.mark.parametrize(
    "kind,machine,target,output,expected",
    [
        ("pwsh", "arm64", "arm64", "7.6.6", True),
        ("pwsh", "x64", "arm64", "7.6.6", False),
        ("pwsh", "x64", "x64", "7.1.0", False),
        ("git", "x64", "x64", "git version 2.51.0.windows.1", True),
        ("git", "arm64", "x64", "git version 2.51.0", False),
        ("git", "x64", "x64", "git version 2.20.0", False),
        ("python", "arm64", "arm64", "3.13.16 win-arm64", True),
        ("python", "arm64", "arm64", "3.13.17 win-arm64", True),
        ("python", "arm64", "arm64", "3.13.15 win-arm64", False),
        ("python", "arm64", "arm64", "3.12.10 win-arm64", False),
        ("python", "arm64", "arm64", "3.14.8 win-arm64", False),
        ("python", "x64", "arm64", "3.13.16 win-amd64", False),
        ("perl", "x64", "arm64", "perl-ok", True),
        ("perl", "arm64", "arm64", "perl-ok", True),
        ("perl", "x64", "arm64", "", False),
        ("protobuf", "x64", "arm64", "libprotoc 33.0", True),
        ("protobuf", "arm64", "arm64", "libprotoc 29.3", True),
        ("protobuf", "x64", "x64", "libprotoc 3.20.0", False),
    ],
)
def test_tool_compatibility(tmp_path, kind, machine, target, output, expected):
    tool = tmp_path / "bin" / "candidate.exe"
    pe_file(tool, machine)
    includes = tmp_path / "include" / "google" / "protobuf"
    includes.mkdir(parents=True)
    (includes / "struct.proto").touch()
    result = run_ps(
        tmp_path,
        f"""
Set-StrictMode -Version Latest
. {literal(FIND_TOOLS)}
function Invoke-ToolProbe {{ return {literal(output)} }}
$result = Test-ToolCandidate '{kind}' {literal(tool)} '{target}'
if ($result -ne ${str(expected).lower()}) {{ throw 'Incorrect compatibility result' }}
""",
    )
    assert result.returncode == 0, result.stderr


def test_protoc_requires_standard_include_tree(tmp_path):
    tool = tmp_path / "bin" / "protoc.exe"
    pe_file(tool, "x64")
    result = run_ps(
        tmp_path,
        f"""
. {literal(FIND_TOOLS)}
function Invoke-ToolProbe {{ return 'libprotoc 33.0' }}
if (Test-ToolCandidate 'protobuf' {literal(tool)} 'arm64') {{ throw 'Missing includes accepted' }}
""",
    )
    assert result.returncode == 0, result.stderr


def test_find_tool_skips_incompatible_candidates(tmp_path):
    wrong = tmp_path / "old" / "pwsh.exe"
    right = tmp_path / "native" / "pwsh.exe"
    pe_file(wrong, "x64")
    pe_file(right, "arm64")
    result = run_ps(
        tmp_path,
        f"""
. {literal(FIND_TOOLS)}
function Invoke-ToolProbe {{ return '7.4.13' }}
$path = Find-Tool 'pwsh' '' @({literal(wrong)}, {literal(right)}) 'arm64'
if ($path -ne {literal(right)}) {{ throw 'Did not select compatible candidate' }}
""",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    sys.platform != "win32", reason="Exercises Windows file access failures"
)
@pytest.mark.parametrize("failure", ["locked", "missing", "access-denied"])
@pytest.mark.parametrize("has_alternative", [True, False])
def test_find_tool_skips_unreadable_executables(tmp_path, failure, has_alternative):
    unreadable = tmp_path / "WindowsApps" / "python.exe"
    alternative = tmp_path / "installed" / "python.exe"
    if failure == "locked":
        pe_file(unreadable, "arm64")
    elif failure == "access-denied":
        unreadable.mkdir(parents=True)
    pe_file(alternative, "arm64")
    result = run_ps(
        tmp_path,
        f"""
Set-StrictMode -Version Latest
. {literal(FIND_TOOLS)}
function Get-ToolCandidates {{
    # Simulate enumeration before the file became inaccessible/disappeared.
    return @({literal(unreadable)}{", " + literal(alternative) if has_alternative else ""})
}}
function Invoke-ToolProbe {{
    param($File, $Arguments)
    if ($File -eq {literal(unreadable)}) {{ throw 'Must not execute an unreadable alias' }}
    return '3.13.16 win-arm64'
}}
$lock = $null
try {{
    if ('{failure}' -eq 'locked') {{
        $lock = [IO.File]::Open({literal(unreadable)}, 'Open', 'ReadWrite', 'None')
    }}
    $path = Find-Tool 'python' 'python.exe' @() 'arm64'
    if ([bool]$path -ne ${str(has_alternative).lower()}) {{ throw 'Incorrect discovery result' }}
    if ($path -and $path -ne {literal(alternative)}) {{ throw 'Did not select the readable candidate' }}
}} finally {{
    if ($lock) {{ $lock.Dispose() }}
}}
""",
    )
    assert result.returncode == 0, result.stderr
    assert "Ignoring unreadable executable" in result.stdout
    assert str(unreadable) in result.stdout


def test_pe_reader_releases_file_handle(tmp_path):
    executable = tmp_path / "python.exe"
    pe_file(executable, "arm64")
    result = run_ps(
        tmp_path,
        f"""
. {literal(FIND_TOOLS)}
if ((Get-PeArchitecture {literal(executable)}) -ne 'arm64') {{ throw 'Wrong architecture' }}
$exclusive = [IO.File]::Open({literal(executable)}, 'Open', 'ReadWrite', 'None')
$exclusive.Dispose()
""",
    )
    assert result.returncode == 0, result.stderr


def test_failed_probe_is_reported_not_accepted(tmp_path):
    result = run_ps(
        tmp_path,
        f"""
. {literal(FIND_TOOLS)}
$value = Invoke-ToolProbe {literal(sys.executable)} @('-c', 'import sys; sys.exit(7)')
if ($null -ne $value) {{ throw 'Failed command accepted' }}
""",
    )
    assert result.returncode == 0, result.stderr
    assert "probe exited 7" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Uses Windows registry discovery")
@pytest.mark.parametrize("missing", ["none", "library", "compiler", "wrong-version"])
def test_cuda_requires_matching_version_and_complete_libraries(tmp_path, missing):
    cuda = tmp_path / "CUDA"
    for relative in [
        "bin/nvcc.exe",
        "bin/ptxas.exe",
        "include/cuda.h",
        *[
            f"lib/arm64/{name}.lib"
            for name in [
                "cuda",
                "cudart",
                "cudart_static",
                "cublas",
                "cublasLt",
                "nvrtc",
                "cufftw",
                "nvml",
                "OpenCL",
            ]
        ],
    ]:
        file = cuda / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    if missing == "library":
        (cuda / "lib" / "arm64" / "cublasLt.lib").unlink()
    if missing == "compiler":
        (cuda / "bin" / "ptxas.exe").unlink()
    result = run_ps(
        tmp_path,
        f"""
. {literal(FIND_TOOLS)}
$env:ProgramFiles = {literal(tmp_path)}
$env:CUDA_PATH = ''
function Invoke-ToolProbe {{ return 'CUDA release {"13.0" if missing == "wrong-version" else "13.4"}' }}
$cuda = Find-CudaToolkit 'arm64' @({literal(cuda)})
if ([bool]$cuda -ne ${str(missing == "none").lower()}) {{ throw 'Wrong CUDA detection' }}
""",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("missing", ["none", "cargo", "wrong-target", "old-rust"])
def test_rust_discovery_resolves_proxy_and_requires_matching_pair(tmp_path, missing):
    proxy = tmp_path / "proxy" / "rustc.exe"
    rust = tmp_path / "native-rust"
    pe_file(proxy, "arm64")
    pe_file(rust / "bin" / "rustc.exe", "arm64")
    if missing != "cargo":
        pe_file(rust / "bin" / "cargo.exe", "arm64")
    result = run_ps(
        tmp_path,
        f"""
. {literal(FIND_TOOLS)}
function Get-ToolCandidates {{ return {literal(proxy)} }}
function Invoke-ToolProbe {{
    param($File, $Arguments)
    if ($Arguments -contains 'sysroot') {{ return {literal(rust)} }}
    if ($File -like '*cargo.exe') {{ return 'cargo 1.95.0 (hash)' }}
    return "host: {"x86_64" if missing == "wrong-target" else "aarch64"}-pc-windows-msvc`nrelease: {"1.94.0" if missing == "old-rust" else "1.95.0"}"
}}
$pair = Find-RustToolchain 'arm64' @({literal(proxy)})
if ([bool]$pair -ne ${str(missing == "none").lower()}) {{ throw 'Wrong Rust detection' }}
if ($pair -and $pair.RustcExecutable -eq {literal(proxy)}) {{ throw 'Saved profile-dependent rustup proxy' }}
""",
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("missing", ["none", "cuda", "rust", "sdk"])
def test_msvc_installs_only_missing_components(tmp_path, missing):
    output = tmp_path / "components.json"
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
$script:installed = $false
function Get-VisualStudioPaths {{ return @('existing-cuda-vs', 'existing-rust-vs') }}
function Test-WindowsSdk {{ return $script:installed -or '{missing}' -ne 'sdk' }}
function Get-InstalledMsvcVersion {{
    param($VisualStudio, $Family, $Target, [switch]$Optional)
    if ($script:installed -or
        ($Family -eq '14.51' -and $VisualStudio -eq 'existing-cuda-vs' -and '{missing}' -ne 'cuda') -or
        ($Family -eq '14.44' -and $VisualStudio -eq 'existing-rust-vs' -and '{missing}' -ne 'rust')) {{
        return "$Family.12345"
    }}
    return $null
}}
function Get-ToolchainPayload {{ return 'unused-installer.exe' }}
function Invoke-ToolchainInstaller {{
    param($File, $Arguments)
    if ('{missing}' -eq 'none') {{ throw 'Unnecessary MSVC installation' }}
    $script:installed = $true
    $Arguments | ConvertTo-Json | Set-Content {literal(output)}
}}
$parameters = Install-MsvcToolchains {literal(tmp_path)} {literal(tmp_path)} 'arm64'
if ('{missing}' -ne 'cuda' -and $parameters.VisualStudioPath -ne 'existing-cuda-vs') {{
    throw 'Replaced existing CUDA compiler'
}}
if ('{missing}' -ne 'rust' -and $parameters.RustVisualStudioPath -ne 'existing-rust-vs') {{
    throw 'Replaced existing Rust compiler'
}}
""",
    )
    assert result.returncode == 0, result.stderr
    if missing == "none":
        assert not output.exists()
    else:
        arguments = json.loads(output.read_text())
        assert arguments.count("--add") == 1
        component = arguments[arguments.index("--add") + 1]
        assert (
            component
            == {
                "cuda": "Microsoft.VisualStudio.Component.VC.Tools.ARM64",
                "rust": "Microsoft.VisualStudio.Component.VC.14.44.17.14.ARM64",
                "sdk": "Microsoft.VisualStudio.Component.Windows11SDK.26100",
            }[missing]
        )


@pytest.mark.parametrize(
    "identity", ["marker", "legacy-config", "unmanaged", "wrong-arch"]
)
def test_toolchain_root_resume_rules(tmp_path, identity):
    root = tmp_path / "tools"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("do not touch")
    if identity != "unmanaged":
        name = (
            "toolchains.json"
            if identity == "legacy-config"
            else ".vllm-toolchains.json"
        )
        (root / name).write_text(
            json.dumps(
                {
                    "architecture": "arm64" if identity == "wrong-arch" else "x64",
                    "parameters": {"PerlPath": "previous-perl"},
                }
            )
        )
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
function Set-ToolchainDirectoryPermissions {{ throw 'Do not change existing root ACLs' }}
$previous = Initialize-ToolchainRoot {literal(root)} 'x64'
""",
    )
    assert (result.returncode == 0) == (identity in ("marker", "legacy-config")), (
        result.stderr
    )
    assert sentinel.read_text() == "do not touch"


@pytest.mark.skipif(sys.platform != "win32", reason="Uses Windows installer root paths")
@pytest.mark.parametrize("architecture", ["x64", "arm64"])
@pytest.mark.parametrize(
    "missing", ["none", "pwsh", "git", "perl", "protobuf", "python", "rust", "cuda"]
)
def test_only_missing_tool_installed_and_rerun_downloads_nothing(
    tmp_path, architecture, missing
):
    root = tmp_path / "managed"
    system = tmp_path / "existing"
    cuda = system / "cuda"
    output = tmp_path / "downloads.json"
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
Set-StrictMode -Version Latest
$env:ProgramFiles = {literal(tmp_path)}
${{env:ProgramFiles(x86)}} = {literal(tmp_path)}
$script:downloads = [Collections.Generic.List[string]]::new()
$script:available = @{{}}
$system = {literal(system)}
$cuda = {literal(cuda)}
function Touch-File {{
    param($Path, $Content = '')
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $Path))
    [IO.File]::WriteAllText($Path, $Content)
}}
foreach ($kind in @('pwsh','git','perl','protobuf','python','rust','cuda')) {{
    $script:available[$kind] = '{missing}' -ne $kind
}}
$executables = @{{pwsh='pwsh.exe';git='git.exe';perl='perl.exe';protobuf='bin\\protoc.exe';python='python.exe'}}
foreach ($kind in $executables.Keys) {{ Touch-File (Join-Path $system $executables[$kind]) }}
Touch-File (Join-Path $system 'include\\google\\protobuf\\struct.proto')
Touch-File (Join-Path $system 'rustc.exe')
Touch-File (Join-Path $system 'cargo.exe')
Touch-File (Join-Path $system 'vs\\VC\\Auxiliary\\Build\\vcvarsall.bat')
Touch-File (Join-Path ${{env:ProgramFiles(x86)}} 'Windows Kits\\10\\Include\\10.0.26100.0\\um\\Windows.h')
Touch-File (Join-Path $cuda 'bin\\nvcc.exe')
Touch-File (Join-Path $cuda 'lib\\{architecture}\\cudart.lib')
Touch-File (Join-Path $cuda 'include\\cuda.h') '#define TENSOR_MAP_ALIGN 64'
function Assert-ToolchainInstallerHost {{}}
function Set-ToolchainDirectoryPermissions {{}}
function Get-ItemProperty {{ return [pscustomobject]@{{LongPathsEnabled=1}} }}
function Get-MachinePython {{ return $null }}
function Test-ToolCandidate {{ return $true }}
function Find-Tool {{
    param($Kind, $Name, $Paths, $Target)
    if (-not $script:available[$Kind]) {{ return $null }}
    if ($Paths[0] -and (Test-Path -LiteralPath $Paths[0])) {{ return $Paths[0] }}
    return (Join-Path $system $executables[$Kind])
}}
function Find-CudaToolkit {{ if ($script:available.cuda) {{ return $cuda }} }}
function Find-RustToolchain {{
    if ($script:available.rust) {{
        return @{{RustcExecutable=(Join-Path $system 'rustc.exe');CargoExecutable=(Join-Path $system 'cargo.exe')}}
    }}
}}
function Install-MsvcToolchains {{
    return @{{VisualStudioPath=(Join-Path $system 'vs');RustVisualStudioPath=(Join-Path $system 'vs');
        MsvcToolsetVersion='14.51.36231';RustMsvcToolsetVersion='14.44.35207';WindowsSdkVersion='10.0.26100.0'}}
}}
function Get-ToolchainPayload {{
    param($Url, $Path)
    $script:downloads.Add($Url)
    return $Path
}}
function Expand-Archive {{
    param($LiteralPath, $DestinationPath, [switch]$Force)
    $kind = [IO.Path]::GetFileNameWithoutExtension($LiteralPath)
    if ($kind -ne '{missing}') {{ throw 'Extracting a tool that is already installed' }}
    $script:available[$kind] = $true
    $relative = @{{pwsh='pwsh.exe';git='cmd\\git.exe';perl='perl\\bin\\perl.exe';protobuf='bin\\protoc.exe'}}[$kind]
    Touch-File (Join-Path $DestinationPath $relative)
    if ($kind -eq 'protobuf') {{ Touch-File (Join-Path $DestinationPath 'include\\google\\protobuf\\struct.proto') }}
}}
function Invoke-ToolchainInstaller {{
    param($File, $Arguments)
    if ($File -like '*python-*.exe' -and '{missing}' -eq 'python') {{
        $script:available.python = $true
        $destination = ($Arguments | Where-Object {{ $_ -like 'TargetDir=*' }}).Substring(10)
        Touch-File (Join-Path $destination 'python.exe')
    }} elseif ($File -like '*rustup-init.exe' -and '{missing}' -eq 'rust') {{
        $script:available.rust = $true
    }} elseif ('{missing}' -eq 'cuda' -and -not $script:available.cuda) {{
        $script:available.cuda = $true
    }} else {{ throw 'Unnecessary installer invocation' }}
}}
function Invoke-CheckedCommand {{
    param($File, $Arguments, [switch]$Capture)
    if ($File -like '*nvcc.exe') {{ return 'CUDA release {"13.4" if architecture == "arm64" else "13.0"}' }}
}}
[void](Install-WindowsToolchains '{architecture}' {literal(root)} $cuda)
$count = $script:downloads.Count
function Invoke-ToolchainInstaller {{ throw 'Rerun must not run installers' }}
function Expand-Archive {{ throw 'Rerun must not extract archives' }}
[void](Install-WindowsToolchains '{architecture}' {literal(root)} $cuda)
if ($script:downloads.Count -ne $count) {{ throw 'Rerun downloaded tools again' }}
ConvertTo-Json -InputObject @($script:downloads) | Set-Content {literal(output)}
""",
    )
    assert result.returncode == 0, result.stderr
    downloads = json.loads(output.read_text())
    assert len(downloads) == (0 if missing == "none" else 1)
    if missing == "cuda":
        assert downloads == [
            "https://developer.download.nvidia.com/compute/cuda/13.4.2/local_installers/cuda_13.4.2_windows_arm64.exe"
            if architecture == "arm64"
            else "https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_windows.exe"
        ]
    config = json.loads((root / "toolchains.json").read_text(encoding="utf-8-sig"))
    assert config["architecture"] == architecture
    for key in [
        "PythonExecutable",
        "PerlPath",
        "ProtocPath",
        "CargoExecutable",
        "RustcExecutable",
    ]:
        assert Path(config["parameters"][key]).is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="Probes native Windows executables")
def test_actual_python_and_powershell_are_recognized(tmp_path):
    result = run_ps(
        tmp_path,
        f"""
. {literal(FIND_TOOLS)}
$target = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString().ToLowerInvariant()
if (-not (Test-ToolCandidate 'pwsh' {literal(PWSH)} $target)) {{ throw 'Failed native pwsh probe' }}
$pythonTarget = if ((Get-PeArchitecture {literal(sys.executable)}) -eq 'arm64') {{'arm64'}} else {{'x64'}}
if (-not (Test-ToolCandidate 'python' {literal(sys.executable)} $pythonTarget '{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')) {{
    throw 'Failed native Python probe'
}}
""",
    )
    assert result.returncode == 0, result.stderr


def test_cached_download_is_verified_without_network(tmp_path):
    payload = tmp_path / "cached.zip"
    payload.write_bytes(b"existing-download")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
function Invoke-WebRequest {{ throw 'Cached payload must not be downloaded again' }}
$file = Get-ToolchainPayload 'https://example.invalid/cached.zip' {literal(payload)} '{digest}'
if ($file -ne {literal(payload)}) {{ throw 'Wrong payload path' }}
""",
    )
    assert result.returncode == 0, result.stderr


def test_reboot_request_is_persisted(tmp_path):
    marker = tmp_path / ".reboot-required"
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
$script:ToolchainRebootMarker = {literal(marker)}
function Get-ToolchainBootIdentity {{ return 'boot-1' }}
function Start-Process {{ return [pscustomobject]@{{ExitCode=3010}} }}
Invoke-ToolchainInstaller 'never-executed.exe' @('/quiet')
if (-not $script:ToolchainRebootRequired) {{ throw 'Lost reboot requirement' }}
""",
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8-sig").strip() == "boot-1"


@pytest.mark.parametrize("missing", ["none", "linker", "headers", "libraries"])
def test_msvc_requires_complete_target_toolset(tmp_path, missing):
    toolset = tmp_path / "VC" / "Tools" / "MSVC" / "14.44.35207"
    files = [
        "bin/Hostx64/x64/cl.exe",
        "bin/Hostx64/x64/link.exe",
        "include/vcruntime.h",
        "lib/x64/libcmt.lib",
    ]
    for relative in files:
        file = toolset / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    bad_file = {"linker": files[1], "headers": files[2], "libraries": files[3]}.get(
        missing
    )
    if bad_file:
        (toolset / bad_file).unlink()
    result = run_ps(
        tmp_path,
        f"""
. {literal(TOOLCHAINS)}
$version = Get-InstalledMsvcVersion {literal(tmp_path)} '14.44' 'x64' -Optional
if ([bool]$version -ne ${str(missing == "none").lower()}) {{ throw 'Incorrect MSVC completeness detection' }}
""",
    )
    assert result.returncode == 0, result.stderr
