# Machine-level helpers, loaded by provision-pool.ps1. Compatible with PowerShell 5.1.
. (Join-Path $PSScriptRoot "find-toolchains.ps1")
$script:ToolchainRebootMarker = $null

function Get-ToolchainBootIdentity {
    return (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString("o")
}

function Assert-ToolchainInstallerHost {
    param([string]$Target)
    if ($env:OS -ne "Windows_NT") {
        throw "Toolchain installation requires Windows."
    }
    $osArch = Get-ItemPropertyValue `
        "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" `
        -Name PROCESSOR_ARCHITECTURE
    $expected = if ($Target -eq "x64") { "AMD64" } else { "ARM64" }
    if ($osArch -ine $expected) {
        throw "This machine is $osArch, not Windows $Target."
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run -InstallToolchains in an elevated Windows PowerShell or PowerShell 7 terminal."
    }
}

function Assert-Payload {
    param([string]$Path, [string]$Sha256, [string]$Publisher)
    if ($Sha256) {
        if ($Sha256 -notmatch '^[a-fA-F0-9]{64}$' -or
            (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ine $Sha256) {
            throw "SHA-256 verification failed: $Path"
        }
    } elseif ($Publisher) {
        $signature = Get-AuthenticodeSignature -LiteralPath $Path
        if ($signature.Status -ne "Valid" -or
            $signature.SignerCertificate.Subject -notmatch [regex]::Escape($Publisher)) {
            throw "Expected a valid $Publisher Authenticode signature: $Path"
        }
    } else {
        throw "A checksum or publisher is required before executing or extracting $Path."
    }
}

function Get-ToolchainPayload {
    param([string]$Url, [string]$Path, [string]$Sha256 = "", [string]$Publisher = "")
    if (-not $Url.StartsWith("https://", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Toolchain downloads require HTTPS: $Url"
    }
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "Downloading $Url"
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Path -TimeoutSec 1800
    }
    Assert-Payload $Path $Sha256 $Publisher
    return $Path
}

function Invoke-ToolchainInstaller {
    param([string]$File, [string[]]$Arguments)
    # Start-Process joins ArgumentList without quoting; preserve spaces in paths.
    $quoted = $Arguments | ForEach-Object {
        '"' + (($_ -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
    }
    $process = Start-Process -FilePath $File -ArgumentList ($quoted -join " ") -Wait -PassThru
    if ($process.ExitCode -eq 3010) {
        $script:ToolchainRebootRequired = $true
        if ($script:ToolchainRebootMarker) {
            Get-ToolchainBootIdentity | Set-Content -LiteralPath $script:ToolchainRebootMarker -Encoding UTF8
        }
        Write-Warning "$File requires a reboot before pool preparation."
    } elseif ($process.ExitCode -ne 0) {
        throw "$File failed with exit code $($process.ExitCode). Check the vendor installer log."
    }
}

function Set-ToolchainDirectoryPermissions {
    param([string]$Path)
    $acl = [Security.AccessControl.DirectorySecurity]::new()
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($sid in @("S-1-5-18", "S-1-5-32-544", "S-1-5-32-545")) {
        $rights = if ($sid -eq "S-1-5-32-545") { "ReadAndExecute" } else { "FullControl" }
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
            [Security.Principal.SecurityIdentifier]::new($sid),
            $rights, "ContainerInherit,ObjectInherit", "None", "Allow"
        ))
    }
    Set-Acl -LiteralPath $Path -AclObject $acl
}

function Get-InstalledMsvcVersion {
    param([string]$VisualStudio, [string]$Family, [string]$Target, [switch]$Optional)
    $toolsets = Join-Path $VisualStudio "VC\Tools\MSVC"
    $versions = @()
    if (Test-Path -LiteralPath $toolsets) {
        $versions = @(Get-ChildItem -LiteralPath $toolsets -Directory |
            Where-Object { $_.Name -match ("^" + [regex]::Escape($Family) + '\.\d+$') } |
            Sort-Object { [version]$_.Name } -Descending)
    }
    foreach ($version in $versions) {
        if (-not (Test-Path -LiteralPath (Join-Path $version.FullName "include\vcruntime.h")) -or
            -not (Test-Path -LiteralPath (Join-Path $version.FullName "lib\$Target\libcmt.lib"))) { continue }
        $hosts = if ($Target -eq "arm64") { @("HostARM64", "Hostx64") } else { @("Hostx64") }
        foreach ($hostArch in $hosts) {
            $bin = Join-Path $version.FullName "bin\$hostArch\$Target"
            if ((Test-Path -LiteralPath (Join-Path $bin "cl.exe")) -and
                (Test-Path -LiteralPath (Join-Path $bin "link.exe"))) {
                return $version.Name
            }
        }
    }
    if ($Optional) { return $null }
    throw "Visual Studio did not install an MSVC $Family compiler targeting $Target under $toolsets."
}

function Get-MachinePython {
    param([string]$Target, [string]$Version)
    $minor = ($Version.Split(".")[0..1] -join ".")
    $keys = if ($Target -eq "arm64") { @("$minor-arm64", $minor) } else { @($minor) }
    foreach ($key in $keys) {
        $registry = "HKLM:\SOFTWARE\Python\PythonCore\$key\InstallPath"
        if (-not (Test-Path -LiteralPath $registry)) { continue }
        $registration = Get-ItemProperty -LiteralPath $registry
        $property = $registration.PSObject.Properties["ExecutablePath"]
        if (-not $property -or -not (Test-Path -LiteralPath $property.Value -PathType Leaf)) {
            continue
        }
        if (Test-ToolCandidate "python" $property.Value $Target $Version) {
            return $property.Value
        }
    }
    return $null
}

function Install-MsvcToolchains {
    param([string]$Root, [string]$Downloads, [string]$Target, [hashtable]$ExistingTools = @{})
    $cudaFamily = if ($Target -eq "arm64") { "14.51" } else { "14.44" }
    $cudaVs = $null
    $rustVs = $null
    $cudaVersion = $null
    $rustVersion = $null
    foreach ($candidate in (Get-VisualStudioPaths $Root $ExistingTools)) {
        if (-not $cudaVs) {
            $cudaVersion = Get-InstalledMsvcVersion $candidate $cudaFamily $Target -Optional
            if ($cudaVersion) { $cudaVs = $candidate }
        }
        if (-not $rustVs) {
            $rustVersion = Get-InstalledMsvcVersion $candidate "14.44" $Target -Optional
            if ($rustVersion) { $rustVs = $candidate }
        }
    }
    $sdkInstalled = Test-WindowsSdk $Target
    if ($cudaVs) { Write-Host "Reusing CUDA compiler MSVC $cudaVersion at $cudaVs" }
    if ($rustVs) { Write-Host "Reusing Rust/OpenSSL compiler MSVC $rustVersion at $rustVs" }
    $components = @()
    if (-not $sdkInstalled) { $components += "Microsoft.VisualStudio.Component.Windows11SDK.26100" }
    if ($Target -eq "arm64") {
        if (-not $cudaVs) { $components += "Microsoft.VisualStudio.Component.VC.Tools.ARM64" }
        if (-not $rustVs) { $components += "Microsoft.VisualStudio.Component.VC.14.44.17.14.ARM64" }
    } elseif (-not $cudaVs) {
        $components += "Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64"
    }
    if ($components.Count) {
        $major = if ($Target -eq "arm64") { "18" } else { "17" }
        $vs = Join-Path $Root "vs$major"
        # VS 18.6.3 carries MSVC 14.51; a moving Stable channel could select 14.52.
        $url = if ($Target -eq "arm64") {
            "https://download.visualstudio.microsoft.com/download/pr/471731ab-b194-4597-af69-194c13cb642c/3fb30c58cf04776a188dd7c6480f0a9b1a6f74202e755223947a33bfbf23d133/vs_BuildTools.exe"
        } else {
            "https://aka.ms/vs/17/release/vs_BuildTools.exe"
        }
        $bootstrapper = Get-ToolchainPayload `
            $url (Join-Path $Downloads "vs_BuildTools.exe") -Publisher "Microsoft Corporation"
        $arguments = @("--quiet", "--wait", "--norestart", "--nocache", "--installPath", $vs)
        if (Test-Path -LiteralPath (Join-Path $vs "Common7\Tools\Launch-VsDevShell.ps1")) {
            $arguments = @("modify") + $arguments
        }
        foreach ($component in $components) { $arguments += @("--add", $component) }
        Invoke-ToolchainInstaller $bootstrapper $arguments
        if (-not $cudaVs) {
            $cudaVersion = Get-InstalledMsvcVersion $vs $cudaFamily $Target
            $cudaVs = $vs
        }
        if (-not $rustVs) {
            $rustVersion = Get-InstalledMsvcVersion $vs "14.44" $Target
            $rustVs = $vs
        }
        if (-not (Test-WindowsSdk $Target)) {
            throw "Windows SDK 10.0.26100.0 for $Target is incomplete after installation."
        }
    }
    return @{
        VisualStudioPath = $cudaVs
        RustVisualStudioPath = $rustVs
        MsvcToolsetVersion = $cudaVersion
        RustMsvcToolsetVersion = $rustVersion
        WindowsSdkVersion = "10.0.26100.0"
    }
}

function Initialize-ToolchainRoot {
    param([string]$Root, [string]$Target)
    $marker = Join-Path $Root ".vllm-toolchains.json"
    $configPath = Join-Path $Root "toolchains.json"
    $previous = @{}
    if (Test-Path -LiteralPath $Root) {
        $identityPath = if (Test-Path -LiteralPath $marker) { $marker } else { $configPath }
        if (-not (Test-Path -LiteralPath $identityPath)) {
            throw "Existing ToolchainRoot is not a managed toolchain directory: $Root"
        }
        $identity = Get-Content -LiteralPath $identityPath -Raw | ConvertFrom-Json
        if ($identity.architecture -ne $Target) { throw "ToolchainRoot does not target $Target." }
        if (Test-Path -LiteralPath $configPath) {
            $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
            if ($config.architecture -ne $Target) { throw "Recorded toolchains do not target $Target." }
            foreach ($entry in $config.parameters.PSObject.Properties) { $previous[$entry.Name] = $entry.Value }
        }
    } else {
        New-Item -ItemType Directory -Path $Root | Out-Null
        Set-ToolchainDirectoryPermissions $Root
    }
    @{architecture = $Target} | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding UTF8
    return $previous
}

function Repair-Cuda13Header {
    param([string]$Cuda, [string]$Backup)
    $header = Join-Path $Cuda "include\cuda.h"
    $content = [IO.File]::ReadAllText($header)
    if ($content -notmatch "TENSOR_MAP_ALIGN") {
        $declaration = "typedef struct CUtensorMap_st {"
        if (-not $content.Contains($declaration) -or
            -not $content.Contains("alignas(128)") -or -not $content.Contains("_Alignas(128)")) {
            throw "Unrecognized CUDA 13.0 tensor-map header; refusing to patch it."
        }
        $content = $content.Replace($declaration, @"
#if defined(_MSC_VER)
#define TENSOR_MAP_ALIGN 64
#else
#define TENSOR_MAP_ALIGN 128
#endif
$declaration
"@).Replace("alignas(128)", "alignas(TENSOR_MAP_ALIGN)").Replace(
            "_Alignas(128)", "_Alignas(TENSOR_MAP_ALIGN)"
        )
        [IO.File]::Copy($header, $Backup, $false)
        [IO.File]::WriteAllText($header, $content, [Text.UTF8Encoding]::new($false))
    }
}

function Install-WindowsToolchains {
    param(
        [string]$Target,
        [string]$Root,
        [string]$RequestedCudaPath,
        [string]$PythonVersion = "3.13.16",
        [hashtable]$ExistingTools = @{}
    )
    Assert-ToolchainInstallerHost $Target
    if ($Root -notmatch '^[a-zA-Z]:\\' -or $Root -match '[\r\n"]') {
        throw "ToolchainRoot must be an absolute local Windows directory."
    }
    $previous = Initialize-ToolchainRoot $Root $Target
    foreach ($entry in $ExistingTools.GetEnumerator()) { $previous[$entry.Key] = $entry.Value }
    $cuda = Find-CudaToolkit $Target @($RequestedCudaPath, $previous["CudaPath"])
    $script:ToolchainRebootMarker = Join-Path $Root ".reboot-required"
    $script:ToolchainRebootRequired = (Test-Path -LiteralPath $script:ToolchainRebootMarker) -and
        (Get-Content -LiteralPath $script:ToolchainRebootMarker -Raw).Trim() -eq (Get-ToolchainBootIdentity)
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $downloads = Join-Path $Root "downloads"
    if (-not (Test-Path -LiteralPath $downloads)) { New-Item -ItemType Directory -Path $downloads | Out-Null }
    $isArm = $Target -eq "arm64"
    $hostTriple = if ($isArm) { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
    $pythonArch = if ($isArm) { "arm64" } else { "amd64" }
    $gitArch = if ($isArm) { "arm64" } else { "64-bit" }
    $psHash = if ($isArm) {
        "1820febe6f9567c8bab21be601dacb902777c1185e1beb81843c3a6f902d6b9d"
    } else {
        "8fb52d2172d285b230c2857a90ba4dd28ecf6477ba4a91f91b6854a647b33b65"
    }
    $gitHash = if ($isArm) {
        "b21755ccd10f71a37ec341ca9ac450cebee71bb1e70c0d88d90ddd6e5b16dfa4"
    } else {
        "c2c955a21fa99889d83f485f24fa5d9a38fffc2d509d4022385510e11c26b250"
    }
    $rustupHash = if ($isArm) {
        "de9f7d29ccd39efa59a3dda3ec363b396e09b92681229b9b8f6aaa4c84285e9c"
    } else {
        "88d8258dcf6ae4f7a80c7d1088e1f36fa7025a1cfd1343731b4ee6f385121fc0"
    }
    $archives = @(
        @{
            Name = "pwsh"; Hash = $psHash
            Key = "PwshExecutable"; Executable = "pwsh.exe"
            Paths = @($previous["PwshExecutable"], (Join-Path $Root "pwsh\pwsh.exe"),
                (Join-Path $env:ProgramFiles "PowerShell\7\pwsh.exe"))
            Url = "https://github.com/PowerShell/PowerShell/releases/download/v7.4.13/PowerShell-7.4.13-win-$Target.zip"
        },
        @{
            Name = "git"; Hash = $gitHash
            Key = "GitExecutable"; Executable = "cmd\git.exe"
            Paths = @($previous["GitExecutable"], (Join-Path $Root "git\cmd\git.exe"),
                (Join-Path $env:ProgramFiles "Git\cmd\git.exe"))
            Url = "https://github.com/git-for-windows/git/releases/download/v2.51.0.windows.1/MinGit-2.51.0-$gitArch.zip"
        },
        @{
            Name = "perl"; Hash = "6a081a811781c30aca51dbc036afd93092af91e3297901f02c17043795a10690"
            Key = "PerlPath"; Executable = "perl\bin\perl.exe"
            Paths = @($previous["PerlPath"], (Join-Path $Root "perl\perl\bin\perl.exe"),
                "C:\Strawberry\perl\bin\perl.exe")
            Url = "https://github.com/StrawberryPerl/Perl-Dist-Strawberry/releases/download/SP_54231_64bit/strawberry-perl-5.42.3.1-64bit-portable.zip"
        },
        @{
            Name = "protobuf"; Hash = "3742cd49c8b6bd78b6760540367eb0ff62fa70a1032e15dafe131bfaf296986a"
            Key = "ProtocPath"; Executable = "bin\protoc.exe"
            Paths = @($previous["ProtocPath"], (Join-Path $Root "protobuf\bin\protoc.exe"))
            Url = "https://github.com/protocolbuffers/protobuf/releases/download/v33.0/protoc-33.0-win64.zip"
        }
    )
    $parameters = @{}
    foreach ($package in $archives) {
        $include = ""
        # An explicit include tree belongs only to its paired protoc, not a PATH fallback.
        $executable = $null
        if ($package.Name -eq "protobuf" -and $previous["ProtocPath"] -and $previous["ProtocIncludePath"]) {
            $executable = Find-Tool "protobuf" "" @($previous["ProtocPath"]) $Target `
                -ProtocInclude $previous["ProtocIncludePath"]
            if ($executable) { $include = $previous["ProtocIncludePath"] }
        }
        if (-not $executable) {
            $executable = Find-Tool $package.Name ([IO.Path]::GetFileName($package.Executable)) $package.Paths $Target
        }
        if (-not $executable) {
            Write-Host "Installing missing compatible $($package.Name)"
            $archive = Get-ToolchainPayload $package.Url (Join-Path $downloads "$($package.Name).zip") $package.Hash
            Expand-Archive -LiteralPath $archive -DestinationPath (Join-Path $Root $package.Name) -Force
            $executable = Join-Path (Join-Path $Root $package.Name) $package.Executable
            if (-not (Test-ToolCandidate $package.Name $executable $Target)) {
                throw "Installed $($package.Name) failed validation: $executable"
            }
        }
        $parameters[$package.Key] = $executable
        if ($package.Name -eq "protobuf") {
            $parameters.ProtocIncludePath = if ($include) { $include } else {
                Join-Path (Split-Path -Parent (Split-Path -Parent $executable)) "include"
            }
        }
    }
    if ($isArm) {
        Write-Host "Perl and protoc may use x64 emulation; compilers and Python target native ARM64."
    }
    $pythonMinor = ([version]$PythonVersion).ToString(2)
    $pythonHome = Join-Path $Root "python-$pythonMinor"
    $pythonExe = Find-Tool "python" "python.exe" @($previous["PythonExecutable"],
        (Join-Path $pythonHome "python.exe"), (Join-Path $Root "python\python.exe")) $Target $PythonVersion
    if (-not $pythonExe) { $pythonExe = Get-MachinePython $Target $PythonVersion }
    if (-not $pythonExe) {
        $pythonInstaller = Get-ToolchainPayload `
            "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-$pythonArch.exe" `
            (Join-Path $downloads "python-$PythonVersion-$pythonArch.exe") -Publisher "Python Software Foundation"
        Invoke-ToolchainInstaller $pythonInstaller @(
            "/quiet", "InstallAllUsers=1", "TargetDir=$pythonHome", "PrependPath=0",
            "Include_launcher=0", "Include_test=0", "Include_pip=1"
        )
        $pythonExe = Join-Path $pythonHome "python.exe"
        if (-not (Test-Path -LiteralPath $pythonExe)) {
            # An all-users installer may service an existing installation in place.
            $pythonExe = Get-MachinePython $Target $PythonVersion
        }
        if (-not $pythonExe) { throw "Python installer did not register native Python $PythonVersion." }
    } else {
        Write-Host "Reusing compatible Python at $pythonExe"
    }
    $msvc = Install-MsvcToolchains $Root $downloads $Target $previous
    foreach ($entry in $msvc.GetEnumerator()) { $parameters[$entry.Key] = $entry.Value }
    $cudaVersion = if ($isArm) { "13.4" } else { "13.0" }
    if (-not $cuda) {
        $cudaUrl = if ($isArm) {
            "https://developer.download.nvidia.com/compute/cuda/13.4.2/local_installers/cuda_13.4.2_windows_arm64.exe"
        } else {
            "https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_windows.exe"
        }
        $cudaInstaller = Get-ToolchainPayload $cudaUrl `
            (Join-Path $downloads ([uri]$cudaUrl).Segments[-1]) -Publisher "NVIDIA Corporation"
        Invoke-ToolchainInstaller $cudaInstaller @("-s", "-n")
        $cuda = if ($RequestedCudaPath) { [IO.Path]::GetFullPath($RequestedCudaPath) } else {
            Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v$cudaVersion"
        }
        if (-not (Find-CudaToolkit $Target @($cuda))) { throw "CUDA installation is incomplete: $cuda" }
    }
    $rustBin = Join-Path $Root "rustup\toolchains\1.95.0-$hostTriple\bin"
    $rust = Find-RustToolchain $Target @($previous["RustcExecutable"],
        (Join-Path $rustBin "rustc.exe"))
    if (-not $rust) {
        $rustup = Get-ToolchainPayload `
            "https://static.rust-lang.org/rustup/archive/1.28.2/$hostTriple/rustup-init.exe" `
            (Join-Path $downloads "rustup-init.exe") $rustupHash
        $savedCargo = $env:CARGO_HOME
        $savedRustup = $env:RUSTUP_HOME
        try {
            $env:CARGO_HOME = Join-Path $Root "cargo-bootstrap"
            $env:RUSTUP_HOME = Join-Path $Root "rustup"
            Invoke-ToolchainInstaller $rustup @(
                "-y", "--no-modify-path", "--profile", "minimal",
                "--default-host", $hostTriple, "--default-toolchain", "1.95.0"
            )
        } finally {
            $env:CARGO_HOME = $savedCargo
            $env:RUSTUP_HOME = $savedRustup
        }
        $rust = Find-RustToolchain $Target @((Join-Path $rustBin "rustc.exe"))
        if (-not $rust) { throw "Installed Rust/Cargo failed validation." }
    }
    $parameters.PythonExecutable = $pythonExe
    $parameters.CargoExecutable = $rust.CargoExecutable
    $parameters.RustcExecutable = $rust.RustcExecutable
    $parameters.CudaPath = $cuda
    $files = @(
        $parameters.PythonExecutable, $parameters.GitExecutable, $parameters.PwshExecutable,
        $parameters.CargoExecutable, $parameters.RustcExecutable,
        $parameters.PerlPath, $parameters.ProtocPath,
        (Join-Path $parameters.ProtocIncludePath "google\protobuf\struct.proto"),
        (Join-Path $cuda "bin\nvcc.exe"), (Join-Path $cuda "lib\$Target\cudart.lib"),
        (Join-Path $parameters.VisualStudioPath "VC\Auxiliary\Build\vcvarsall.bat"),
        (Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\Include\10.0.26100.0\um\Windows.h")
    )
    foreach ($file in $files) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
            throw "Installer did not produce the required file: $file"
        }
    }
    $nvccInfo = Invoke-CheckedCommand (Join-Path $cuda "bin\nvcc.exe") @("--version") -Capture
    if ($nvccInfo -notmatch 'release (\d+\.\d+)' -or $Matches[1] -ne $cudaVersion) {
        throw "The installed CUDA toolkit must be version $cudaVersion."
    }
    if (-not (Test-ToolCandidate "python" $parameters.PythonExecutable $Target $PythonVersion)) {
        throw "Python installation must be native $Target with compatible version $PythonVersion."
    }
    Invoke-CheckedCommand $parameters.PerlPath @("-MLocale::Maketext::Simple", "-MIPC::Cmd", "-e", "exit 0")
    Invoke-CheckedCommand $parameters.ProtocPath @("--version")
    if (-not $isArm) {
        Repair-Cuda13Header $cuda (Join-Path $Root "cuda.h.original")
    }
    $longPaths = Get-ItemProperty -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
    if (-not $longPaths.PSObject.Properties["LongPathsEnabled"] -or $longPaths.LongPathsEnabled -ne 1) {
        New-ItemProperty -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
            -Name LongPathsEnabled -Value 1 -PropertyType DWord -Force | Out-Null
    }
    $configPath = Join-Path $Root "toolchains.json"
    [ordered]@{
        architecture = $Target
        reboot_required = $script:ToolchainRebootRequired
        parameters = $parameters
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $configPath -Encoding UTF8
    Write-Host "Toolchains installed. Reboot if requested, then prepare the pool as an unprivileged account:"
    $runtimeArgument = if ($isArm) { "" } else { " -RequirementsFile <runtime-requirements.txt>" }
    Write-Host "& '$($parameters.PwshExecutable)' -NoProfile -File '$PSScriptRoot\provision-pool.ps1' -Architecture $Target -ToolchainConfig '$configPath'$runtimeArgument -CudaArchList <GPU-targets>"
    return $(if ($script:ToolchainRebootRequired) { 3010 } else { 0 })
}
