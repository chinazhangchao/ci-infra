# Windows CUDA CI on private agents

The standalone pipeline in [windows.yml](../../.buildkite/pipelines/windows.yml)
builds **both x64 and arm64** from
[`vortex-captain/vllm-windows`, branch `v029_win_arm64`](https://github.com/vortex-captain/vllm-windows/tree/v029_win_arm64).
It runs directly on Windows, without Docker, WSL, AWS, or Kubernetes.
Existing Linux pipelines and the Linux-oriented pipeline generator are unchanged.

The resolver freezes the branch tip in Buildkite metadata once per build. Both
architectures, including retries, fetch that same SHA into separate temporary
checkouts. x64 builds with MSVC; arm64 invokes the fork's
`tools\build-win-arm64.ps1` rather than duplicating its CUDA overlays, compiler
selection, and Rust/OpenSSL workarounds. Neither job uses a precompiled vLLM.

Each job builds a wheel, checks its platform tags and compiled extension,
installs it into a temporary target **without changing the agent's Python
environment**, and executes the wheel's CUDA `silu_and_mul` kernel against a
PyTorch reference. A missing GPU, incorrect architecture, build failure, or
smoke-test failure fails the job; there is no CPU fallback or configure-only
success. This is build and kernel smoke coverage, not the full Linux test suite
or an end-to-end model inference test.

## Provision the private pools

[`provision-pool.ps1`](provision-pool.ps1) supports the entire setup in two stages:
**install only missing compatible machine tools as Administrator**, then **prepare the pool as
an unprivileged agent account**. This separation keeps CI jobs from inheriting
administrator privileges. Neither stage starts an agent, creates a service,
or saves an agent token.
Use a normal native PowerShell session, not a Visual Studio Developer shell;
the build scripts initialize the compiler environment themselves.

| Tool | Reused when detected | Installed only if no compatible tool is found |
|------|----------------------|----------------------------------------------|
| PowerShell | Native architecture, version 7.2+ | Native PowerShell 7.4.13 (portable) |
| Git | Native architecture, version 2.35+ | Native MinGit 2.51.0 (portable) |
| Python | Native architecture, requested major/minor, patch version at least `-PythonVersion`, with venv/ensurepip | Python 3.13.16 by default; configurable with `-PythonVersion` |
| MSVC | Target compiler, linker, headers and libraries: x64 14.44; ARM64 14.51 for CUDA and 14.44 for Rust/OpenSSL | Only missing compiler components, using VS 2022 or VS 2026 18.6.3 |
| Windows SDK | SDK 10.0.26100.0 headers, target libraries and resource compiler | Only the missing SDK component |
| CUDA | x64 13.0 / ARM64 13.4, `nvcc`, `ptxas`, header and all required target libraries | NVIDIA's public CUDA 13.0 installer for x64; CUDA 13.4.2 installer for ARM64 |
| Rust / Cargo | Native MSVC host, Rust and Cargo 1.95+, complete pair resolved from the actual toolchain rather than a user-specific rustup proxy | Native Rust/Cargo 1.95.0 |
| Perl | Working Perl with `Locale::Maketext::Simple` and `IPC::Cmd` | Full Strawberry Perl 5.42.3.1 portable distribution |
| `protoc` | Version 29+, working compiler and standard `google\protobuf\struct.proto` include tree | Protobuf compiler 33.0 and standard includes |

Discovery checks explicit tool paths, paths recorded by a previous successful
run, the managed toolchain directory, PATH and standard installation locations.
Python also uses machine registration, CUDA checks its registration/environment,
and Visual Studio uses `vswhere` to find installed editions. MSVC compiler
families can be reused from **different** Visual Studio installations. The log
identifies reused tools and rejected incompatible candidates.
Both paths can also use the same installation when it contains both required
toolset versions. Missing compilers are installed under the managed toolchain
root rather than added to an existing Visual Studio installation elsewhere.

Both compiler installations must include `VC\Auxiliary\Build\vcvarsall.bat`.
The legacy MSVC 14.44 component and `VC.CoreBuildTools` alone do not supply
that script. When it is missing, the installer adds the current target tools
component (`VC.Tools.ARM64` or `VC.Tools.x86.x64`) together with
`Microsoft.VisualStudio.Component.VC.CoreBuildTools`. This may install current
compiler support in the managed Rust installation even when the CUDA compiler
is reused from another installation.
If pool preparation reports a missing script in `VisualStudioPath` or
`RustVisualStudioPath`, rerun the toolchain stage as Administrator with the same
root (for example, `.\buildkite\windows\provision-pool.ps1 -Architecture arm64
-InstallToolchains -ToolchainRoot 'C:\vllm-tools\arm64'`). This repairs missing
components and regenerates `toolchains.json`; then retry pool preparation as
the unprivileged agent account. Do not copy scripts between installations or
point the Rust path at a CUDA-only installation without MSVC 14.44.

Unreadable executables, including Windows App Execution Aliases such as
`AppData\Local\Microsoft\WindowsApps\python.exe`, are warned about and skipped.
Discovery continues to another compatible installation or installs the missing
tool; you do not need to disable the aliases.

No download or installer is run for a compatible tool. An older, wrong-architecture
or incomplete installation does not qualify; the script installs a compatible
copy without uninstalling the existing one. For tools in nonstandard locations,
pass the existing executable/path parameters with `-InstallToolchains`, such as
`-PerlPath`, `-ProtocPath`, `-ProtocIncludePath`, `-RustcExecutable`,
`-VisualStudioPath`, or `-RustVisualStudioPath`. Reused paths must be accessible
to the unprivileged agent account, not only the administrator.

The downloaded Perl and `protoc` packages are x64 **host utilities**, including
on ARM64 where Windows x64 emulation is required; existing compatible native
ARM64 tools are also accepted. Python, Rust, Git, PowerShell and the generated
vLLM binaries use the native architecture. Strawberry's bundled MinGW compiler
is not added to PATH: vLLM and OpenSSL use MSVC.

The pool-preparation stage installs **CMake, Ninja and the Python build tools**
from [requirements-toolchain.txt](requirements-toolchain.txt), plus your
runtime dependencies into an isolated venv. ARM64 uses the checked-in
[`requirements-arm64.txt`](requirements-arm64.txt) by default; you do not need
to create a requirements file. Keep `-Wheelhouse` pointing to your existing
Windows ARM64 wheel directory for packages unavailable from public indexes. It verifies
native Python, PyTorch and actual GPU execution, installs the native Buildkite
agent with checksum verification, and generates a launcher.
For **ARM64**, it selects the latest compatible Torch wheel (including
development/nightly versions) from
[`https://pypi.nvidia.cn/nvtorch_oot_nightly/`](https://pypi.nvidia.cn/nvtorch_oot_nightly/).
This index currently publishes Windows ARM64 wheels only; **x64 Torch
installation remains controlled by the runtime manifest/wheelhouse**.
Both Windows architectures default to **Python 3.13**. The pool creates
`<InstallRoot>\venv` using the selected interpreter, without system-site
packages, and checks that it is a real venv with the expected Python version
and architecture **before** installing dependencies. Builds and smoke tests use
that venv's `Scripts\python.exe`; you do not need to activate it manually.

Create two self-hosted Buildkite queues, defaulting to `windows-x64` and
`windows-arm64`. Give this pipeline access to those queues in your private
cluster. Queue names can be changed with `WINDOWS_X64_QUEUE` and
`WINDOWS_ARM64_QUEUE` **on the pipeline upload step**.

Both pools need:

- Native Windows and an NVIDIA GPU/driver supported by their CUDA toolkit.
  ARM64 means native Windows ARM64, not an x64 Python running under emulation.
- Buildkite agent, Git, Python 3.13 (`python` on `PATH`), and PowerShell 7
  (`pwsh` on `PATH`). Configure the agent's command shell as
  `pwsh.exe -NoLogo -NoProfile -NonInteractive -Command`.
- A provisioned, architecture-matched Python environment containing CUDA
  PyTorch, `pip`, `build`, and all the fork's build **and runtime** dependencies.
  The job does not install dependencies from public indexes or replace Torch.
  Pin dependencies in your pool image/private wheelhouse.
- Visual Studio C++ build tools, the appropriate Windows SDK, Rust/Cargo,
  a full Perl distribution for vendored OpenSSL, and Protobuf `protoc` with its
  standard includes. Allow the fork's CMake/Rust dependency downloads, or
  provision their caches. Use a short writable build root, such as `C:\b`,
  with adequate disk space for fresh checkouts and compilation.
- Windows long paths enabled by the pool administrator. Git long-path support
  is enabled only in each job's temporary source checkout.

**x64:** The automated toolchain stage uses the fork's Windows README's Torch/CUDA/compiler
combination (currently Torch `2.11.0+cu130` and CUDA 13). Provision from
`requirements\build\cuda.txt`, `requirements\cuda.txt`, and
`requirements\windows.txt`, resolving all native dependencies for x64. If the
toolkit is selected with `-InstallToolchains`, the script also applies the
fork's MSVC tensor-map alignment fix to CUDA 13.0's `include\cuda.h`, retaining
the original as `cuda.h.original` in the toolchain root. It refuses unrecognized
header layouts rather than silently patching them. The build uses `--no-isolation` and
`--skip-dependency-check`, as the ARM64 helper does: the fork's generic
`pyproject.toml` Torch pin differs from its Windows requirements.

**arm64:** The helper on the selected branch requires **Windows ARM64 PyTorch
built for CUDA 13.4** and the ARM64 CUDA toolkit/libraries. Provisioning selects
Torch from NVIDIA's nightly index automatically; x64 wheels are not substitutes.
The default runtime manifest includes the fork's common requirements and
Windows dependencies, without its x64-only packages. At the time of integration, the
helper defaults to MSVC `14.51.36231` for CUDA, MSVC `14.44.35207` for
Rust/OpenSSL, SDK `10.0.26100.0`, and Rust 1.95+. Its own prerequisite checks
remain authoritative. It also needs environment-local CMake and Ninja.

Set the following in the **agent service environment** or a trusted agent
environment hook (do not check credentials into this repository):

| Variable | Required value |
|----------|----------------|
| `VLLM_WINDOWS_VENV` | Absolute path to the provisioned native venv/Conda environment; must contain `Scripts\python.exe` or `python.exe` |
| `CUDA_PATH` | Matching CUDA toolkit root (`bin\nvcc.exe` and `lib\x64` or `lib\arm64`) |
| `TORCH_CUDA_ARCH_LIST` | GPU targets for that pool, e.g. `8.9` on an appropriate x64 GPU, or `12.0+PTX;10.3a` for the ARM64 helper's default GPU profile |
| `VLLM_WINDOWS_WORK_ROOT` | Short, writable directory for unique disposable source/build/install directories, e.g. `C:\b` |
| `VLLM_WINDOWS_VCVARSALL` | x64 only: full path to the installed `VC\Auxiliary\Build\vcvarsall.bat` |
| `CMAKE_CUDA_ARCHITECTURES` | arm64 only: CMake targets matching `TORCH_CUDA_ARCH_LIST`, e.g. `120-real;103-real` |
| `MAX_JOBS` | Optional compile parallelism, 1-256, default `8` |

For ARM64, optional environment overrides map directly to the fork helper:

| Variable | Helper parameter |
|----------|------------------|
| `VLLM_WINDOWS_VS_PATH` | `VisualStudioPath` |
| `VLLM_WINDOWS_MSVC_VERSION` | `MsvcToolsetVersion` |
| `VLLM_WINDOWS_RUST_VS_PATH` | `RustVisualStudioPath` |
| `VLLM_WINDOWS_RUST_MSVC_VERSION` | `RustMsvcToolsetVersion` |
| `VLLM_WINDOWS_SDK_VERSION` | `WindowsSdkVersion` |
| `VLLM_WINDOWS_PERL_PATH` | `PerlPath` |
| `VLLM_WINDOWS_PROTOC_PATH` | `ProtocPath` |
| `VLLM_WINDOWS_PROTOC_INCLUDE_PATH` | `ProtocIncludePath` |
| `VLLM_WINDOWS_VERSION_OVERRIDE` | `VersionOverride` |

Use isolated/ephemeral agents with clean ci-infra checkouts. Existing
`artifacts\windows-<architecture>` directories cause an error rather than
allowing stale wheels to pass a retry. The Buildkite checkout clean must run
between jobs. Compilation scratch space is unique per invocation and is
removed afterward; wheels and provenance remain in the checkout for upload.
Do not place the provisioned venv inside a Buildkite checkout.

### Run the provisioning script

#### 1. Install missing machine toolchains (Administrator)

Start an **elevated Windows PowerShell 5.1 or PowerShell 7** terminal. No
preinstalled Git, Python, Perl, Protobuf, PowerShell 7, or package manager is
required; download/extract this repository first if Git is not installed.

For **x64**:

```powershell
.\buildkite\windows\provision-pool.ps1 `
    -Architecture x64 -InstallToolchains `
    -ToolchainRoot 'C:\vllm-tools\x64'
```

For **ARM64**:

```powershell
.\buildkite\windows\provision-pool.ps1 `
    -Architecture arm64 -InstallToolchains `
    -ToolchainRoot 'C:\vllm-tools\arm64'
```

When compatible CUDA is missing or incomplete, the script downloads the
architecture-specific local installer directly from NVIDIA. ARM64 uses the
pinned [CUDA 13.4.2 Windows ARM64 installer](https://developer.download.nvidia.com/compute/cuda/13.4.2/local_installers/cuda_13.4.2_windows_arm64.exe)
listed on [NVIDIA's download page](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=arm64&target_version=11&target_type=exe_local).
No installer-path or checksum parameters are needed. The script validates
NVIDIA's Authenticode signature before running it with `-s -n` (silent,
no automatic reboot), and caches downloads by version and architecture.
Existing compatible CUDA 13.4 installations are reused without downloading
or upgrading to 13.4.2. The default expected ARM64 CUDA destination is
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4`; use `-CudaPath`
to discover an existing installation in another directory. This is not an
installer destination override. The ARM64 PyTorch/runtime wheel requirements
below are unchanged.

Both architectures receive missing Perl and `protoc` automatically; no `-PerlPath`
or `-ProtocPath` is needed for standard/PATH installations. Public archive downloads have pinned
SHA-256 checksums. Python, Visual Studio and the public CUDA executable are
checked for valid vendor signatures before execution. Rustup verifies the Rust
toolchain downloads. Download cache entries are reverified before use; a
checksum/signature mismatch stops with the cached file path so you can remove
that specific bad download and retry.
Visual Studio component IDs and installed compiler
families are checked; ARM64 does not silently substitute a newer compiler
family for the fork's required 14.51/14.44 combination. Installed servicing
versions are recorded and passed to the fork helper.

This stage installs missing machine software and enables Windows long paths
only when needed. It does not run the CUDA installer or change drivers when a
complete matching CUDA toolkit is present; GPU/driver functionality is checked
in pool preparation. When CUDA installation is needed it can install/update
the NVIDIA driver. Vendor installers may update their own
registry/PATH settings. It never reboots automatically: exit code **3010**
means **reboot before the next stage**. Installation can take substantial time,
disk space and network bandwidth. The toolchain directory is writable only by
Administrators/SYSTEM and readable/executable by local users. Do not run this
on an active CI worker; drain it first.

#### 2. Prepare the pool (dedicated unprivileged agent account)

Use the installed native PowerShell to prepare a new pool. This stage does not
require elevation. Its directory ACL permits only the agent account, SYSTEM,
and Administrators. `-ToolchainConfig` supplies all installed executable,
compiler and include paths, including Perl and Protobuf; no manual PATH
changes are needed.

After switching from the earlier Python 3.12 setup, rerun `-InstallToolchains`
with the same `ToolchainRoot` to refresh `toolchains.json`. New Python installs
use a minor-version-specific directory (for example, `python-3.13`) and a
version/architecture-specific installer cache, leaving the old Python 3.12
directory and download untouched. An old 3.12 configuration is rejected by
pool preparation instead of silently creating a 3.12 venv. If an `InstallRoot`
already contains a previous pool/venv, choose a new root rather than replacing
its Python in place.

**Only `-RequirementsFile` can be omitted for the ARM64 default. Keep
`-Wheelhouse` for your Windows ARM64 dependency wheels.** The script uses
[`requirements-arm64.txt`](requirements-arm64.txt) and passes your wheelhouse
to pip as `--find-links` for both build and runtime dependencies. Public indexes
remain available unless you also specify `-NoIndex`; Torch is selected
separately from NVIDIA. The manifest is based on the fork's
[`bd3dc3d` requirements](https://github.com/vortex-captain/vllm-windows/tree/bd3dc3d3364a50012bf70b7376010c1b2c89633c/requirements).
It includes common runtime, server, tokenizer, structured-output, Windows event
loop and Triton dependencies. Numba 0.68.0, NVTX 0.2.16, fastsafetensors 0.4.0,
and triton-windows 3.8.0.post29 replace older fork pins to use releases with
Python 3.13 Windows ARM64 wheels.

Some other dependencies (including tiktoken and outlines-core)
do not currently publish matching ARM64 wheels. Supply compatible builds in
your wheelhouse rather than relying on public package availability. Without
a matching wheel, pip may attempt a source build, which can still fail if
the package does not support Windows ARM64. Provisioning does not suppress
such failures. OpenCV headless and the `mistral-common[image]` extra that pulls
it in are omitted from this CI profile; OpenCV-dependent image/video features
require a separate optional installation. TileLang and
InstantTensor are optional acceleration/model-loading backends with no public
Windows ARM64 wheels and are omitted from this build/kernel-smoke CI profile.
The fork's x64-only dependencies are also excluded. This is not a claim of
coverage for every model or optional serving backend.

Use `-RequirementsFile` only to **override** the ARM64 default for a specialized
environment. Omit legacy Torch pins and wheel URLs from overrides; incompatible
pins fail rather than replacing the selected NVIDIA Torch.
For x64, the existing explicit runtime manifest remains required, including its
matching CUDA PyTorch wheel; this ARM64 default does not change the x64 setup.
Use **Python 3.13-compatible wheels** (`cp313`, or an applicable `abi3`/pure
Python wheel) for the matching Windows architecture. Python 3.12-only (`cp312`)
wheels cannot be installed into this venv.

ARM64 Torch selection runs inside the new venv with pip's `--pre`,
`--ignore-installed`, `--no-deps`, and `--only-binary=:all:` options. This is a
**dry run** to choose the latest wheel for that Python/architecture, not an
installation without dependencies. The chosen wheel's version, NVIDIA URL, and
SHA-256 are recorded in `torch-selection.json`; `torch-constraints.txt` then
locks dependency installation to that exact URL/hash and version. Dependencies
are resolved normally, so missing dependencies or incompatible runtime pins
fail provisioning rather than downgrading Torch or falling back to PyPI.
If the manifest includes `torchvision`, `torchaudio`, or other native Torch
extensions, supply versions compatible with the selected Torch build; their
sources are still controlled by your manifest. A nightly Torch build can expose
vLLM build incompatibilities, which must be resolved rather than hidden by a
version fallback.

"Latest" is evaluated when preparing a **new pool**, not at every CI job.
Existing venvs are not upgraded in place. Create a new `InstallRoot` to refresh
Torch, leaving running agents undisturbed.

The script installs `build`, CMake, Ninja, setuptools, setuptools-scm,
setuptools-rust, wheel, packaging, Jinja2, regex, and the Python protobuf package
from its own build-requirements file. Your manifest can further constrain those
versions. With `-NoIndex`, your wheelhouse must contain **both** build and runtime
dependency wheels (except the automatically selected ARM64 Torch wheel).
**On ARM64, `-NoIndex` applies only to the remaining dependency installation**:
Torch selection still contacts NVIDIA's index and the selected wheel URL is
downloaded. This mode is not fully offline. Index settings, if needed, go in your
trusted manifest: pip runs with `--isolated`, ignoring user pip configuration
and `PIP_*` environment variables. `-NoIndex` disables package-index lookup;
direct URL references in a manifest are still honored by pip. For disconnected
dependency installation, use only local references and a complete wheelhouse.
The agent release download still needs access to GitHub.

Example **x64** invocation:

```powershell
$tools = Get-Content 'C:\vllm-tools\x64\toolchains.json' -Raw | ConvertFrom-Json
& $tools.parameters.PwshExecutable -NoProfile `
    -File .\buildkite\windows\provision-pool.ps1 `
    -Architecture x64 `
    -ToolchainConfig 'C:\vllm-tools\x64\toolchains.json' `
    -RequirementsFile 'C:\pool-inputs\requirements-x64.txt' `
    -Wheelhouse 'C:\pool-inputs\wheels-x64' -NoIndex `
    -CudaArchList '8.9' `
    -InstallRoot 'C:\bk\x64'
```

Example **ARM64** invocation:

```powershell
$tools = Get-Content 'C:\vllm-tools\arm64\toolchains.json' -Raw | ConvertFrom-Json
& $tools.parameters.PwshExecutable -NoProfile `
    -File .\buildkite\windows\provision-pool.ps1 `
    -Architecture arm64 `
    -ToolchainConfig 'C:\vllm-tools\arm64\toolchains.json' `
    -Wheelhouse 'C:\pool-inputs\wheels-arm64' `
    -CudaArchList '12.0+PTX;10.3a' `
    -CMakeCudaArchitectures '120-real;103-real' `
    -InstallRoot 'C:\bk\arm64'
```

Replace the wheelhouse path with your actual directory. Add `-NoIndex` only
if it contains the complete build/runtime dependency set; otherwise pip can
use your local ARM64 wheels alongside public packages. pip selects compatible
versions across both sources, so wheelhouse versions must satisfy the manifest.
The NVIDIA Torch selection and exact-wheel constraint remain unchanged.

For already provisioned toolchains, the original explicit
`-PythonExecutable`, `-CudaPath`, `-VisualStudioPath`, `-PerlPath`, `-ProtocPath`,
and `-ProtocIncludePath` parameters remain supported **instead of**
`-ToolchainConfig`. In this reuse mode, Git and Rust must be on PATH or specified
with `-GitExecutable`, `-CargoExecutable`, and `-RustcExecutable`; PowerShell
must already be native 7.2+. The ARM64 compiler/SDK overrides remain available.
Do not combine manual tool paths with `-ToolchainConfig`: its recorded values
are authoritative.

Use `-Queue` for a custom queue and `-MaxJobs` for compilation concurrency.
Buildkite defaults to release `4.3.0`;
`-AgentVersion` selects another explicit stable release, never a moving `latest`.

On success, the root contains `venv`, `bin\buildkite-agent.exe`,
`buildkite-agent.cfg`, `environment.json`, `provisioning.json`,
`start-agent.ps1`, `checkouts`, and `work`. The launcher sets the per-agent
environment shown above, so you do not need to set it manually. It preserves
the agent's exit code and restores the calling process environment on exit.
ARM64 pools also retain `torch-selection.json` and `torch-constraints.txt` for
the selected NVIDIA Torch version, source URL, and checksum.

Supply a cluster agent token **at startup**, not during provisioning. For
example, have your secret manager place it in a protected file readable only
by the agent account and administrators, then run:

```powershell
$env:BUILDKITE_AGENT_TOKEN = 'file://C:\secrets\buildkite-agent-token'
& 'C:\bk\x64\start-agent.ps1'  # Use C:\bk\arm64 on the ARM64 host.
```

The launcher stays in the foreground. For unattended operation, configure your
existing service supervisor to run it with native `pwsh.exe -NoProfile -File`
under the **same account**, supplying the token through its secret mechanism.
The script deliberately does not create a LocalSystem service or persist a
token in generated files. Confirm the agent appears in the intended Buildkite
queue after starting it.

Rerun `-InstallToolchains` with the **same ToolchainRoot** to reuse validated
tools and finish missing components after a partial run. A managed root has
`.vllm-toolchains.json` (or a `toolchains.json` produced by the earlier script);
unrelated directories and roots for a different architecture are rejected.
Incomplete managed portable copies can be repaired, but valid ones are not
re-extracted. Existing system Visual Studio instances are inspected, not modified;
missing components are added to the script's managed VS instance. Reboot requests
are remembered across retries until the machine has rebooted.

Pool preparation still requires a **new InstallRoot**, preserving environment
isolation and avoiding changes to live agents. Machine installers can service
existing machine-level products when a compatible installation is absent.
A failed rerun preserves the last successful `toolchains.json`; a first failed
run writes no completion configuration. Pool failures write no
`provisioning.json`, and the launcher refuses incomplete pools.
A successful provision establishes dependency/tool availability,
not a successful vLLM compilation; run the Windows pipeline to establish that.

## Create and run the Buildkite pipeline

Create a Buildkite pipeline with **this ci-infra repository** as its repository,
not the vLLM fork. Set its initial steps to:

```yaml
steps:
  - label: ":pipeline: Upload Windows CUDA pipeline"
    agents:
      queue: windows-x64
    command: 'buildkite-agent pipeline upload .buildkite\pipelines\windows.yml'
```

For custom queue names, change the initial step's queue and set
`WINDOWS_X64_QUEUE` / `WINDOWS_ARM64_QUEUE` in its environment before upload.
No upstream vLLM Buildkite queues or credentials are needed.

Run manually, on a schedule, or trigger this pipeline from your fork's trusted
automation. The default builds the latest `v029_win_arm64` once resolved.
To reproduce a specific source revision, set `VLLM_WINDOWS_COMMIT` to a full
40-character fork SHA when creating the build. `VLLM_WINDOWS_BRANCH` can
override the default branch for a new build. These are separate from
`BUILDKITE_COMMIT` / `BUILDKITE_BRANCH`, which identify **ci-infra**.
Changing source variables on a resolver retry does not change its saved SHA;
start a new build instead.

This repository does not install a webhook or post commit statuses to the
separately cloned fork. To gate fork PRs, trusted automation must pass their
exact SHA and report this pipeline's result back to that SHA.

Never automatically run untrusted fork PR code on persistent privileged
private agents. Restrict who can trigger builds and change source variables;
use manual approval and disposable agents for reviewed external changes.
The fork's build scripts run native code on the agent. Keep pool service
accounts unprivileged and avoid production credentials.

## Outputs and local checks

Each architecture uploads `artifacts/windows-<architecture>/**/*`: its wheel,
SHA-256 checksum, source/ci-infra commit provenance, Python/Torch runtime
metadata, and `smoke.json` on success. Compilation and failure output is in the
Buildkite job log. A wheel may be uploaded for diagnosis after a failed smoke
test; only consume artifacts from a successful job.

Run infrastructure tests (no CUDA needed):

```powershell
python -m pytest buildkite\tests\test_windows_ci.py buildkite\tests\test_windows_provisioning.py buildkite\tests\test_windows_tool_discovery.py
```

Native `.cmd` exit-code tests run on Windows and are skipped on other hosts.
Provisioning tests require PowerShell 7; they mock tool installation and agent
startup and never register an agent or install CUDA/toolchains.
Actual wheel compilation and CUDA kernel execution require the provisioned
private pools; infrastructure tests do not establish GPU compatibility.
