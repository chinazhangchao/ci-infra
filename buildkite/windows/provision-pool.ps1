#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet("x64", "arm64")][string]$Architecture,
    [switch]$InstallToolchains,
    [string]$ToolchainRoot = "",
    [string]$ToolchainConfig = "",
    [ValidatePattern('^3\.(10|11|12|13|14)\.\d+$')][string]$PythonVersion = "3.13.16",
    [string]$PythonExecutable = "",
    [string]$CudaPath = "",
    [string]$VisualStudioPath = "",
    [string]$RequirementsFile = "",
    [string]$CudaArchList = "",
    [string]$GitExecutable = "git.exe",
    [string]$PwshExecutable = "",
    [string]$CargoExecutable = "cargo.exe",
    [string]$RustcExecutable = "rustc.exe",
    [string]$InstallRoot = "",
    [string]$Wheelhouse = "",
    [switch]$NoIndex,
    [string]$CMakeCudaArchitectures = "",
    [ValidatePattern('^[a-zA-Z0-9._-]+$')][string]$Queue = "",
    [ValidatePattern('^\d+\.\d+\.\d+$')][string]$AgentVersion = "4.3.0",
    [string]$RustVisualStudioPath = "",
    [ValidatePattern('^\d+\.\d+\.\d+$')][string]$MsvcToolsetVersion = "14.51.36231",
    [ValidatePattern('^\d+\.\d+\.\d+$')][string]$RustMsvcToolsetVersion = "14.44.35207",
    [ValidatePattern('^\d+\.\d+\.\d+\.\d+$')][string]$WindowsSdkVersion = "10.0.26100.0",
    [string]$PerlPath = "perl.exe",
    [string]$ProtocPath = "protoc.exe",
    [string]$ProtocIncludePath = "",
    [ValidateRange(1, 256)][int]$MaxJobs = 8
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Import-ToolchainConfig {
    param([string]$Path, [string]$Target)
    $config = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($config.architecture -ne $Target) {
        throw "ToolchainConfig does not target $Target."
    }
    $allowed = @(
        "PythonExecutable", "CudaPath", "VisualStudioPath", "RustVisualStudioPath",
        "MsvcToolsetVersion", "RustMsvcToolsetVersion", "WindowsSdkVersion",
        "GitExecutable", "PwshExecutable", "CargoExecutable", "RustcExecutable",
        "PerlPath", "ProtocPath", "ProtocIncludePath"
    )
    foreach ($entry in $config.parameters.PSObject.Properties) {
        if ($entry.Name -notin $allowed -or $entry.Value -isnot [string]) {
            throw "Unsupported toolchain configuration field: $($entry.Name)"
        }
        Set-Variable -Name $entry.Name -Value $entry.Value -Scope Script
    }
}

function Invoke-CheckedCommand {
    param([string]$File, [string[]]$Arguments, [switch]$Capture)
    if ($Capture) {
        $output = & $File @Arguments
    } else {
        & $File @Arguments | Out-Host
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$File failed with exit code $LASTEXITCODE."
    }
    if ($Capture) {
        return ($output -join "`n")
    }
}

function Select-NvidiaTorch {
    param([string]$Python, [string]$Root)
    $index = "https://pypi.nvidia.cn/nvtorch_oot_nightly/"
    $reportPath = Join-Path $Root "torch-selection.json"
    Invoke-CheckedCommand $Python @(
        "-I", "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
        "--dry-run", "--ignore-installed", "--no-deps", "--pre", "--only-binary=:all:",
        "--index-url", $index, "--report", $reportPath, "torch"
    )
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    $packages = @($report.install)
    if ($packages.Count -ne 1 -or $packages[0].metadata.name -ne "torch") {
        throw "NVIDIA Torch selection must contain exactly one torch wheel."
    }
    $package = $packages[0]
    $version = $package.metadata.version
    $url = [uri]$package.download_info.url
    $hash = $package.download_info.archive_info.hashes.sha256
    if ($version -notmatch '^\d[0-9A-Za-z.!+_-]*$' -or
        $url.Scheme -ne "https" -or $url.Authority -ne "pypi.nvidia.cn" -or
        -not $url.AbsolutePath.StartsWith("/nvtorch_oot_nightly/torch/", [StringComparison]::Ordinal) -or
        -not $url.AbsolutePath.EndsWith("-win_arm64.whl", [StringComparison]::Ordinal) -or
        $hash -notmatch '^[a-fA-F0-9]{64}$') {
        throw "Expected a SHA-256-identified Windows ARM64 Torch wheel from $index."
    }
    $wheel = [UriBuilder]::new($url)
    $wheel.Fragment = "sha256=$hash"
    $constraints = Join-Path $Root "torch-constraints.txt"
    "torch @ $($wheel.Uri.AbsoluteUri)" |
        Set-Content -LiteralPath $constraints -Encoding utf8
    Write-Host "Selected NVIDIA Torch $version. Runtime requirements must not pin an incompatible Torch version."
    return @{ Version = $version; Constraints = $constraints }
}

function Resolve-Tool {
    param([string]$Path)
    $command = Get-Command $Path -CommandType Application -ErrorAction Stop
    return $command.Source
}

function Resolve-RequiredPath {
    param([string]$Path, [ValidateSet("Leaf", "Container")][string]$Type)
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType $Type)) {
        throw "Required $Type not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-VcVars {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Container) {
        $Path = Join-Path $Path "VC\Auxiliary\Build\vcvarsall.bat"
    }
    return Resolve-RequiredPath $Path Leaf
}

function Assert-NativeHost {
    param([string]$Target)
    if (
        -not $IsWindows -or
        [Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() -ine $Target -or
        [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString() -ine $Target
    ) {
        throw "Run this script in native PowerShell 7 on Windows $Target, not under emulation."
    }
    $longPaths = Get-ItemPropertyValue `
        -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
        -Name LongPathsEnabled
    if ($longPaths -ne 1) {
        throw "Ask the pool administrator to enable Windows long paths before provisioning."
    }
}

function Assert-Arm64Toolset {
    param([string]$VcVars, [string]$Version)
    $root = $VcVars
    1..4 | ForEach-Object { $root = Split-Path -Parent $root }
    $toolset = Join-Path $root "VC\Tools\MSVC\$Version"
    $compilers = @("HostARM64", "Hostx64") | ForEach-Object {
        Join-Path $toolset "bin\$_\arm64\cl.exe"
    }
    if (-not ($compilers | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })) {
        throw "ARM64 MSVC compiler $Version was not found under $toolset."
    }
    [void](Resolve-RequiredPath (Join-Path $toolset "lib\arm64") Container)
}

function Protect-PoolDirectory {
    param([string]$Path)
    $acl = [Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    $identities = @(
        [Security.Principal.WindowsIdentity]::GetCurrent().User,
        [Security.Principal.SecurityIdentifier]::new("S-1-5-18"),
        [Security.Principal.SecurityIdentifier]::new("S-1-5-32-544")
    )
    foreach ($identity in $identities) {
        $rule = [Security.AccessControl.FileSystemAccessRule]::new(
            $identity, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
        )
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Install-PoolAgent {
    param([string]$Root, [string]$Target, [string]$Version)
    $releaseArch = if ($Target -eq "x64") { "amd64" } else { "arm64" }
    $filename = "buildkite-agent-windows-$releaseArch-$Version.zip"
    $release = "https://github.com/buildkite/agent/releases/download/v$Version"
    $archivePath = Join-Path $Root $filename
    $sumsPath = Join-Path $Root "buildkite-agent-$Version.SHA256SUMS"
    Invoke-WebRequest "$release/buildkite-agent-$Version.SHA256SUMS" -OutFile $sumsPath -TimeoutSec 300
    Invoke-WebRequest "$release/$filename" -OutFile $archivePath -TimeoutSec 300
    $pattern = '(?m)^([a-fA-F0-9]{64})\s+\*?' + [regex]::Escape($filename) + '\r?$'
    $matches = [regex]::Matches([IO.File]::ReadAllText($sumsPath), $pattern)
    $hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash
    if ($matches.Count -ne 1 -or $hash -ine $matches[0].Groups[1].Value) {
        throw "Buildkite release checksum verification failed for $filename."
    }
    $bin = Join-Path $Root "bin"
    New-Item -ItemType Directory -Path $bin | Out-Null
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $entry = $archive.GetEntry("buildkite-agent.exe")
        if (-not $entry) {
            throw "The release archive contains no buildkite-agent.exe."
        }
        [IO.Compression.ZipFileExtensions]::ExtractToFile(
            $entry, (Join-Path $bin "buildkite-agent.exe"), $false
        )
    } finally {
        $archive.Dispose()
    }
    Invoke-CheckedCommand (Join-Path $bin "buildkite-agent.exe") @("--version")
    Remove-Item -LiteralPath $archivePath, $sumsPath
    return $hash
}

function Invoke-PoolProvisioning {
    if ($PSVersionTable.PSVersion -lt [version]"7.2") {
        throw "Use PowerShell 7.2+ for pool preparation. Windows PowerShell 5.1 supports -InstallToolchains only."
    }
    if ($ToolchainConfig) {
        Import-ToolchainConfig $ToolchainConfig $Architecture
    }
    if (-not $RequirementsFile -and $Architecture -eq "arm64") {
        $RequirementsFile = Join-Path $PSScriptRoot "requirements-arm64.txt"
    }
    if (-not $RequirementsFile -or -not $CudaArchList -or -not $PythonExecutable -or
        -not $CudaPath -or -not $VisualStudioPath) {
        throw "Pool preparation requires RequirementsFile, CudaArchList and either ToolchainConfig or the Python/CUDA/Visual Studio paths."
    }
    Assert-NativeHost $Architecture
    if ($InstallRoot -and -not [IO.Path]::IsPathFullyQualified($InstallRoot)) {
        throw "InstallRoot must be an absolute path outside any source checkout."
    }
    $root = if ($InstallRoot) {
        [IO.Path]::GetFullPath($InstallRoot)
    } else {
        "C:\bk\$Architecture"
    }
    if (Test-Path -LiteralPath $root) {
        throw "InstallRoot already exists: $root. Use a new directory; live pools are never overwritten."
    }
    $requirements = Resolve-RequiredPath $RequirementsFile Leaf
    Write-Host "Using runtime requirements: $requirements"
    $wheels = if ($Wheelhouse) { Resolve-RequiredPath $Wheelhouse Container } else { "" }
    if ($wheels) { Write-Host "Using build/runtime wheelhouse: $wheels" }
    if ($Architecture -eq "arm64" -and -not $CMakeCudaArchitectures) {
        throw "CMakeCudaArchitectures is required for ARM64 and must match CudaArchList."
    }
    $python = Resolve-Tool $PythonExecutable
    $pythonProbe = "import json,sys,sysconfig; print(json.dumps(dict(platform=sysconfig.get_platform(),version=list(sys.version_info[:3]),is_venv=sys.prefix!=sys.base_prefix)))"
    $pythonInfo = Invoke-CheckedCommand $python @("-I", "-c", $pythonProbe) -Capture | ConvertFrom-Json
    $platform = if ($Architecture -eq "x64") { "win-amd64" } else { "win-arm64" }
    $wantedPython = [version]$PythonVersion
    $actualPython = [version]($pythonInfo.version -join ".")
    if (
        $pythonInfo.platform -cne $platform -or
        $actualPython.Major -ne $wantedPython.Major -or
        $actualPython.Minor -ne $wantedPython.Minor -or $actualPython -lt $wantedPython
    ) {
        throw "PythonExecutable must be native $platform Python $($wantedPython.Major).$($wantedPython.Minor), patch $($wantedPython.Build) or newer. Rerun -InstallToolchains to refresh an older ToolchainConfig."
    }
    $cuda = Resolve-RequiredPath $CudaPath Container
    foreach ($file in @("bin\nvcc.exe", "bin\ptxas.exe", "include\cuda.h")) {
        [void](Resolve-RequiredPath (Join-Path $cuda $file) Leaf)
    }
    foreach ($library in @(
        "cuda", "cudart", "cudart_static", "cublas", "cublasLt",
        "nvrtc", "cufftw", "nvml", "OpenCL"
    )) {
        [void](Resolve-RequiredPath (Join-Path $cuda "lib\$Architecture\$library.lib") Leaf)
    }
    $nvcc = Invoke-CheckedCommand (Join-Path $cuda "bin\nvcc.exe") @("--version") -Capture
    if ($nvcc -notmatch 'release (\d+\.\d+)') {
        throw "Unable to determine the CUDA toolkit version from nvcc."
    }
    $cudaVersion = $Matches[1]
    if ($Architecture -eq "arm64" -and $cudaVersion -ne "13.4") {
        throw "The fork's ARM64 helper requires CUDA 13.4."
    }
    $vcvars = Resolve-VcVars $VisualStudioPath
    $rustVcvars = $vcvars
    if ($Architecture -eq "arm64") {
        if ($RustVisualStudioPath) {
            $rustVcvars = Resolve-VcVars $RustVisualStudioPath
        }
        Assert-Arm64Toolset $vcvars $MsvcToolsetVersion
        Assert-Arm64Toolset $rustVcvars $RustMsvcToolsetVersion
        $sdk = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10"
        foreach ($directory in @(
            "Include\$WindowsSdkVersion\ucrt", "Include\$WindowsSdkVersion\um",
            "Lib\$WindowsSdkVersion\ucrt\arm64", "Lib\$WindowsSdkVersion\um\arm64"
        )) {
            [void](Resolve-RequiredPath (Join-Path $sdk $directory) Container)
        }
    }
    $git = Resolve-Tool $GitExecutable
    $pwsh = if ($PwshExecutable) { Resolve-Tool $PwshExecutable } else { Join-Path $PSHOME "pwsh.exe" }
    $cargo = Resolve-Tool $CargoExecutable
    $rustc = Resolve-Tool $RustcExecutable
    $rust = Invoke-CheckedCommand $rustc @("-vV") -Capture
    $rustHost = if ($Architecture -eq "arm64") {
        "aarch64-pc-windows-msvc"
    } else {
        "x86_64-pc-windows-msvc"
    }
    if (
        $rust -notmatch "(?m)^host: $([regex]::Escape($rustHost))\r?$" -or
        $rust -notmatch '(?m)^release: (\d+\.\d+\.\d+)'
    ) {
        throw "Rust must target $rustHost."
    }
    if ([version]$Matches[1] -lt [version]"1.95.0") {
        throw "Rust 1.95 or newer is required."
    }
    $perl = Resolve-Tool $PerlPath
    Invoke-CheckedCommand $perl @("-MLocale::Maketext::Simple", "-MIPC::Cmd", "-e", "exit 0")
    $protoc = Resolve-Tool $ProtocPath
    $protoInclude = if ($ProtocIncludePath) {
        Resolve-RequiredPath $ProtocIncludePath Container
    } else {
        Join-Path (Split-Path -Parent (Split-Path -Parent $protoc)) "include"
    }
    [void](Resolve-RequiredPath (Join-Path $protoInclude "google\protobuf\struct.proto") Leaf)
    Invoke-CheckedCommand $protoc @("--version")

    New-Item -ItemType Directory -Path $root | Out-Null
    Protect-PoolDirectory $root
    $venv = Join-Path $root "venv"
    Invoke-CheckedCommand $python @("-I", "-m", "venv", $venv)
    $venvPython = Join-Path $venv "Scripts\python.exe"
    $venvInfo = Invoke-CheckedCommand $venvPython @("-I", "-c", $pythonProbe) -Capture | ConvertFrom-Json
    if (-not $venvInfo.is_venv -or $venvInfo.platform -cne $platform -or
        ($venvInfo.version -join ".") -ne ($pythonInfo.version -join ".")) {
        throw "The new venv must use the selected native Python $actualPython."
    }
    $buildRequirements = Join-Path $PSScriptRoot "requirements-toolchain.txt"
    $pipArguments = @(
        "-I", "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
        "-r", $buildRequirements, "-r", $requirements
    )
    if ($wheels) { $pipArguments += @("--find-links", $wheels) }
    if ($NoIndex) { $pipArguments += "--no-index" }
    $paths = @(
        (Join-Path $root "bin"), (Join-Path $venv "Scripts"), (Join-Path $cuda "bin"),
        (Split-Path -Parent $git), (Split-Path -Parent $pwsh),
        (Split-Path -Parent $cargo), (Split-Path -Parent $rustc),
        (Split-Path -Parent $perl), (Split-Path -Parent $protoc)
    ) | Select-Object -Unique
    $dependencyEnvironment = @{
        CUDA_PATH = $cuda; CUDA_HOME = $cuda; CUDA_ROOT = $cuda
        PROTOC = $protoc; PROTOC_INCLUDE = $protoInclude
        CARGO_HOME = (Join-Path $root "cargo-cache")
    }
    $originalEnvironment = @{}
    $originalPath = $env:PATH
    $torchSelection = $null
    try {
        $env:PATH = (@($paths) + @($originalPath)) -join [IO.Path]::PathSeparator
        foreach ($entry in $dependencyEnvironment.GetEnumerator()) {
            $originalEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
        if ($Architecture -eq "arm64") {
            Write-Host "Selecting the latest compatible Torch from NVIDIA's nightly index."
            if ($NoIndex) {
                Write-Host "NoIndex applies to the remaining dependencies; Torch selection still uses NVIDIA's index."
            }
            $torchSelection = Select-NvidiaTorch $venvPython $root
            $pipArguments += @("-c", $torchSelection.Constraints, "torch==$($torchSelection.Version)")
        }
        Write-Host "Installing the pool's dependency manifest into $venv"
        Invoke-CheckedCommand $venvPython $pipArguments
        Invoke-CheckedCommand $venvPython @("-I", "-m", "pip", "--isolated", "check")
        foreach ($tool in @("cmake.exe", "ninja.exe")) {
            [void](Resolve-RequiredPath (Join-Path $venv "Scripts\$tool") Leaf)
        }
        $runtime = Invoke-CheckedCommand $venvPython @("-I", "-c", @'
import json, sysconfig
import build, setuptools, setuptools_scm, setuptools_rust, wheel, packaging, jinja2, regex
import google.protobuf
import torch
if not torch.version.cuda or not torch.cuda.is_available():
    raise RuntimeError("CUDA-enabled native PyTorch and a working NVIDIA GPU are required.")
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
print(json.dumps(dict(platform=sysconfig.get_platform(), torch=torch.__version__,
                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))))
'@) -Capture | ConvertFrom-Json
    } finally {
        $env:PATH = $originalPath
        foreach ($entry in $originalEnvironment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
        }
    }
    if ($runtime.platform -cne $platform -or $runtime.cuda -ne $cudaVersion) {
        throw "Installed PyTorch must target $platform and CUDA $cudaVersion."
    }
    $agentHash = Install-PoolAgent $root $Architecture $AgentVersion
    $queueName = if ($Queue) { $Queue } else { "windows-$Architecture" }
    $settings = [ordered]@{
        VLLM_WINDOWS_VENV = $venv
        VLLM_WINDOWS_WORK_ROOT = (Join-Path $root "work")
        CUDA_PATH = $cuda
        CUDA_HOME = $cuda
        CUDA_ROOT = $cuda
        TORCH_CUDA_ARCH_LIST = $CudaArchList
        MAX_JOBS = $MaxJobs.ToString()
        BUILDKITE_BUILD_PATH = (Join-Path $root "checkouts")
        PROTOC = $protoc
        PROTOC_INCLUDE = $protoInclude
        CARGO_HOME = (Join-Path $root "cargo-cache")
    }
    if ($Architecture -eq "x64") {
        $settings.VLLM_WINDOWS_VCVARSALL = $vcvars
    } else {
        $settings.CMAKE_CUDA_ARCHITECTURES = $CMakeCudaArchitectures
        $settings.VLLM_WINDOWS_VS_PATH = $vcvars
        $settings.VLLM_WINDOWS_MSVC_VERSION = $MsvcToolsetVersion
        $settings.VLLM_WINDOWS_RUST_VS_PATH = $rustVcvars
        $settings.VLLM_WINDOWS_RUST_MSVC_VERSION = $RustMsvcToolsetVersion
        $settings.VLLM_WINDOWS_SDK_VERSION = $WindowsSdkVersion
        $settings.VLLM_WINDOWS_PERL_PATH = $perl
        $settings.VLLM_WINDOWS_PROTOC_PATH = $protoc
        $settings.VLLM_WINDOWS_PROTOC_INCLUDE_PATH = $protoInclude
    }
    $configuration = [ordered]@{ environment = $settings; paths = @($paths) }
    $configuration | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $root "environment.json") -Encoding utf8
    @"
name="vllm-$Architecture-%hostname"
queue="$queueName"
shell="pwsh.exe -NoLogo -NoProfile -NonInteractive -Command"
git-clone-flags="-v -c core.longpaths=true"
git-clean-flags="-ffdx"
"@ | Set-Content -LiteralPath (Join-Path $root "buildkite-agent.cfg") -Encoding utf8
    foreach ($directory in @($settings.VLLM_WINDOWS_WORK_ROOT, $settings.BUILDKITE_BUILD_PATH)) {
        New-Item -ItemType Directory -Path $directory | Out-Null
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "start-agent.ps1") -Destination $root
    [ordered]@{
        architecture = $Architecture
        queue = $queueName
        agent_version = $AgentVersion
        agent_sha256 = $agentHash
        requirements_sha256 = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
        torch_selection = $torchSelection
        runtime = $runtime
    } | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $root "provisioning.json") -Encoding utf8
    Write-Host "Pool prepared at $root for queue $queueName; no agent has been started."
    Write-Host "As this account, supply BUILDKITE_AGENT_TOKEN and run: & '$root\start-agent.ps1'"
}

if ($MyInvocation.InvocationName -ne ".") {
    if ($InstallToolchains) {
        if ($ToolchainConfig) { throw "Use InstallToolchains and ToolchainConfig in separate invocations." }
        . (Join-Path $PSScriptRoot "install-toolchains.ps1")
        $toolsRoot = if ($ToolchainRoot) { $ToolchainRoot } else { "C:\vllm-tools\$Architecture" }
        $existing = @{}
        foreach ($name in @("PythonExecutable", "GitExecutable", "PwshExecutable", "RustcExecutable",
            "PerlPath", "ProtocPath", "ProtocIncludePath", "VisualStudioPath", "RustVisualStudioPath")) {
            if ($PSBoundParameters.ContainsKey($name)) { $existing[$name] = $PSBoundParameters[$name] }
        }
        exit (Install-WindowsToolchains $Architecture $toolsRoot $CudaPath $PythonVersion $existing)
    } else {
        Invoke-PoolProvisioning
    }
}
