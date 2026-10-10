# Discovery is read-only. Missing or incompatible candidates are not installation failures.

function Get-ToolCandidates {
    param([string]$Name, [string[]]$Paths)
    $candidates = @()
    foreach ($path in ($Paths | Where-Object { $_ })) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $candidates += (Resolve-Path -LiteralPath $path).Path
        } else {
            $candidates += @(Get-Command $path -CommandType Application -All -ErrorAction SilentlyContinue |
                ForEach-Object { $_.Source })
        }
    }
    if ($Name) {
        $candidates += @(Get-Command $Name -CommandType Application -All -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Source })
    }
    return @($candidates | Where-Object {
        $_ -and (Test-Path -LiteralPath $_ -PathType Leaf)
    } | Select-Object -Unique)
}

function Get-PeArchitecture {
    param([string]$Path)
    $stream = $null
    $reader = $null
    try {
        $stream = [IO.File]::OpenRead($Path)
        $reader = [IO.BinaryReader]::new($stream)
        if ($stream.Length -lt 64 -or $reader.ReadUInt16() -ne 0x5a4d) { return "" }
        $stream.Position = 60
        $offset = $reader.ReadUInt32()
        if ($offset -gt $stream.Length - 6) { return "" }
        $stream.Position = $offset
        if ($reader.ReadUInt32() -ne 0x4550) { return "" }
        switch ($reader.ReadUInt16()) {
            0x8664 { return "x64" }
            0xaa64 { return "arm64" }
            default { return "" }
        }
    } catch [IO.IOException], [UnauthorizedAccessException] {
        # Windows App Execution Aliases can be listed on PATH but cannot be read as PE files.
        Write-Warning "Ignoring unreadable executable ${Path}: $($_.Exception.Message)"
        return ""
    } finally {
        if ($reader) {
            $reader.Dispose()
        } elseif ($stream) {
            $stream.Dispose()
        }
    }
}

function Invoke-ToolProbe {
    param([string]$File, [string[]]$Arguments)
    # A broken candidate is reported and skipped, not mistaken for a working tool.
    $ErrorActionPreference = "Continue"
    $PSNativeCommandUseErrorActionPreference = $false
    try {
        $output = & $File @Arguments 2>&1
        if ($LASTEXITCODE -eq 0) { return ($output -join "`n").Trim() }
        Write-Warning "Ignoring ${File}: probe exited $LASTEXITCODE. $($output -join ' ')"
    } catch [System.Management.Automation.ApplicationFailedException], [System.ComponentModel.Win32Exception] {
        Write-Warning "Ignoring ${File}: $($_.Exception.Message)"
    }
    return $null
}

function Test-ToolCandidate {
    param([string]$Kind, [string]$Path, [string]$Target, [string]$PythonVersion = "3.13.16",
          [string]$ProtocInclude = "")
    $machine = Get-PeArchitecture $Path
    $hostUtility = $Kind -in @("perl", "protobuf")
    if (-not $machine -or (-not $hostUtility -and $machine -ne $Target) -or
        ($Target -eq "x64" -and $machine -eq "arm64")) { return $false }
    switch ($Kind) {
        "pwsh" {
            $info = Invoke-ToolProbe $Path @("-NoLogo", "-NoProfile", "-NonInteractive",
                "-Command", '$PSVersionTable.PSVersion.ToString()')
            return $info -match '^\d+\.\d+\.\d+$' -and [version]$info -ge [version]"7.2.0"
        }
        "git" {
            $info = Invoke-ToolProbe $Path @("--version")
            return $info -match '^git version (\d+\.\d+\.\d+)' -and
                [version]$Matches[1] -ge [version]"2.35.0"
        }
        "python" {
            $info = Invoke-ToolProbe $Path @("-I", "-c",
                "import platform,sysconfig,venv,ensurepip; print(platform.python_version()+' '+sysconfig.get_platform())")
            $tag = if ($Target -eq "arm64") { "win-arm64" } else { "win-amd64" }
            if ($info -notmatch "^(\d+\.\d+\.\d+) $tag$") { return $false }
            $actual = [version]$Matches[1]
            $wanted = [version]$PythonVersion
            return $actual.Major -eq $wanted.Major -and $actual.Minor -eq $wanted.Minor -and $actual -ge $wanted
        }
        "perl" {
            $info = Invoke-ToolProbe $Path @("-MLocale::Maketext::Simple", "-MIPC::Cmd",
                "-e", 'print "perl-ok"')
            return $info -eq "perl-ok"
        }
        "protobuf" {
            if (-not $ProtocInclude) {
                $ProtocInclude = Join-Path (Split-Path -Parent (Split-Path -Parent $Path)) "include"
            }
            if (-not (Test-Path -LiteralPath (Join-Path $ProtocInclude "google\protobuf\struct.proto"))) {
                return $false
            }
            $info = Invoke-ToolProbe $Path @("--version")
            return $info -match '^libprotoc (\d+)\.(\d+)' -and [int]$Matches[1] -ge 29
        }
    }
    throw "Unknown tool kind: $Kind"
}

function Find-Tool {
    param([string]$Kind, [string]$Name, [string[]]$Paths, [string]$Target,
          [string]$PythonVersion = "3.13.16", [string]$ProtocInclude = "")
    foreach ($candidate in (Get-ToolCandidates $Name $Paths)) {
        if (Test-ToolCandidate $Kind $candidate $Target $PythonVersion $ProtocInclude) {
            Write-Host "Reusing $Kind at $candidate"
            return $candidate
        }
        Write-Host "Ignoring incompatible or incomplete $Kind at $candidate"
    }
    return $null
}

function Find-RustToolchain {
    param([string]$Target, [string[]]$Paths)
    $triple = if ($Target -eq "arm64") { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
    foreach ($candidate in (Get-ToolCandidates "rustc.exe" $Paths)) {
        if ((Get-PeArchitecture $candidate) -ne $Target) { continue }
        # Resolve rustup proxies to real binaries, independent of the administrator's profile.
        $sysroot = Invoke-ToolProbe $candidate @("--print", "sysroot")
        if (-not $sysroot -or -not (Test-Path -LiteralPath $sysroot -PathType Container)) { continue }
        $rustc = Join-Path $sysroot "bin\rustc.exe"
        $cargo = Join-Path $sysroot "bin\cargo.exe"
        if (-not (Test-Path -LiteralPath $rustc) -or -not (Test-Path -LiteralPath $cargo)) { continue }
        if ((Get-PeArchitecture $rustc) -ne $Target -or (Get-PeArchitecture $cargo) -ne $Target) { continue }
        $rustInfo = Invoke-ToolProbe $rustc @("-vV")
        if ($rustInfo -notmatch "(?m)^host: $triple\r?$" -or
            $rustInfo -notmatch '(?m)^release: (\d+\.\d+\.\d+)\r?$' -or
            [version]$Matches[1] -lt [version]"1.95.0") { continue }
        $cargoInfo = Invoke-ToolProbe $cargo @("--version")
        if ($cargoInfo -notmatch '^cargo (\d+\.\d+\.\d+) ' -or
            [version]$Matches[1] -lt [version]"1.95.0") { continue }
        Write-Host "Reusing native Rust/Cargo from $sysroot"
        return @{ RustcExecutable = $rustc; CargoExecutable = $cargo }
    }
    return $null
}

function Find-CudaToolkit {
    param([string]$Target, [string[]]$Paths)
    $version = if ($Target -eq "arm64") { "13.4" } else { "13.0" }
    $candidates = @($Paths) + @($env:CUDA_PATH,
        (Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v$version"))
    $registry = "HKLM:\SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA\v$version"
    if (Test-Path -LiteralPath $registry) {
        $property = (Get-ItemProperty -LiteralPath $registry).PSObject.Properties["InstallDir"]
        if ($property) { $candidates += $property.Value }
    }
    foreach ($path in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        $files = @("bin\nvcc.exe", "bin\ptxas.exe", "include\cuda.h")
        $files += @("cuda", "cudart", "cudart_static", "cublas", "cublasLt",
            "nvrtc", "cufftw", "nvml", "OpenCL") | ForEach-Object { "lib\$Target\$_.lib" }
        $missing = @($files | Where-Object { -not (Test-Path -LiteralPath (Join-Path $path $_) -PathType Leaf) })
        if ($missing.Count) { continue }
        $info = Invoke-ToolProbe (Join-Path $path "bin\nvcc.exe") @("--version")
        if ($info -match 'release (\d+\.\d+)' -and $Matches[1] -eq $version) {
            Write-Host "Reusing CUDA $version at $path"
            return $path
        }
    }
    return $null
}

function Test-WindowsSdk {
    param([string]$Target)
    $sdk = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10"
    $files = @(
        "Include\10.0.26100.0\um\Windows.h", "Include\10.0.26100.0\ucrt\stdio.h",
        "Include\10.0.26100.0\shared\sdkddkver.h",
        "Lib\10.0.26100.0\um\$Target\kernel32.lib", "Lib\10.0.26100.0\ucrt\$Target\ucrt.lib",
        "bin\10.0.26100.0\$Target\rc.exe"
    )
    return @($files | Where-Object { -not (Test-Path -LiteralPath (Join-Path $sdk $_) -PathType Leaf) }).Count -eq 0
}

function Get-VisualStudioPaths {
    param([string]$Root, [hashtable]$ExistingTools)
    $paths = @($ExistingTools["VisualStudioPath"], $ExistingTools["RustVisualStudioPath"],
        (Join-Path $Root "vs18"), (Join-Path $Root "vs17"), $env:VSINSTALLDIR)
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $output = Invoke-ToolProbe $vswhere @("-all", "-products", "*", "-property", "installationPath")
        if ($output) { $paths += $output -split "\r?\n" }
    }
    foreach ($path in ($paths | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            1..4 | ForEach-Object { $path = Split-Path -Parent $path }
        }
        if (Test-Path -LiteralPath (Join-Path $path "VC\Auxiliary\Build\vcvarsall.bat")) { $path }
    }
}
