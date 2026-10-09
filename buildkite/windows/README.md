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

For machines with the toolchains already installed, use
[`provision-pool.ps1`](provision-pool.ps1). It prepares a **new** pool directory,
creates an isolated venv, installs your dependency manifest, verifies native
Python/PyTorch and an actual CUDA operation, installs the native Buildkite agent
from a pinned release after checking its published SHA-256 checksum, and
generates a launcher. It does not install CUDA/Visual Studio, change machine-wide
PATH, enable long paths, create a Windows service, start an agent, or save tokens.

Create two self-hosted Buildkite queues, defaulting to `windows-x64` and
`windows-arm64`. Give this pipeline access to those queues in your private
cluster. Queue names can be changed with `WINDOWS_X64_QUEUE` and
`WINDOWS_ARM64_QUEUE` **on the pipeline upload step**.

Both pools need:

- Native Windows and an NVIDIA GPU/driver supported by their CUDA toolkit.
  ARM64 means native Windows ARM64, not an x64 Python running under emulation.
- Buildkite agent, Git, Python 3.10+ (`python` on `PATH`), and PowerShell 7
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

**x64:** Follow the fork's Windows README for the matching Torch/CUDA/compiler
combination (currently Torch `2.11.0+cu130` and CUDA 13). Provision from
`requirements\build\cuda.txt`, `requirements\cuda.txt`, and
`requirements\windows.txt`, resolving all native dependencies for x64. If the
toolkit needs the fork's CUDA alignment fix, apply it while provisioning the
image, not in an unprivileged CI job. The build uses `--no-isolation` and
`--skip-dependency-check`, as the ARM64 helper does: the fork's generic
`pyproject.toml` Torch pin differs from its Windows requirements.

**arm64:** The helper on the selected branch requires a **private Windows ARM64
PyTorch build for CUDA 13.4** and the ARM64 CUDA toolkit/libraries. Public x64
PyTorch wheels are not substitutes. Provision compatible ARM64 build and
runtime dependencies from your own wheelhouse; do not blindly install the
x64-only pins in the generic requirements. At the time of integration, the
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

Run **native PowerShell 7** as the dedicated account that will run the agent.
The account needs permission to create the requested installation directory,
but the script does not require elevation. Windows long paths must already
be enabled. The new directory's ACL permits only this account, SYSTEM, and
Administrators. Do not provision as Administrator and then run CI as
Administrator; provision under the intended unprivileged agent identity.

Prepare a requirements file for each architecture containing **all build and
runtime dependencies**, including the matching CUDA PyTorch wheel. Prefer a
tested, version-pinned manifest and prebuilt dependency wheels. Do not include
vLLM itself. For x64, the fork's three requirements files listed above are the
starting point; its CUDA Torch pins need the matching PyTorch index or your
wheelhouse. For ARM64, provide your private CUDA 13.4 Torch wheel and other
ARM64 dependencies through `-Wheelhouse` or explicit references in the manifest.
The script cannot manufacture these private packages.

The manifest must include `build`, `pip`, CMake, Ninja, setuptools,
setuptools-scm, setuptools-rust, wheel, packaging, Jinja2, regex, and protobuf,
as well as the runtime dependencies. Index settings, if needed, go in that
trusted manifest: pip runs with `--isolated`, ignoring user pip configuration
and `PIP_*` environment variables. `-NoIndex` disables package-index lookup;
direct URL references in a manifest are still honored by pip. For disconnected
dependency installation, use only local references and a complete wheelhouse.
The agent release download still needs access to GitHub.

Example **x64** invocation (replace tool paths and dependency inputs):

```powershell
.\buildkite\windows\provision-pool.ps1 `
    -Architecture x64 `
    -PythonExecutable 'C:\Python312\python.exe' `
    -CudaPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0' `
    -VisualStudioPath 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools' `
    -RequirementsFile 'C:\pool-inputs\requirements-x64.txt' `
    -Wheelhouse 'C:\pool-inputs\wheels-x64' -NoIndex `
    -CudaArchList '8.9' `
    -ProtocPath 'C:\tools\protobuf\bin\protoc.exe' `
    -ProtocIncludePath 'C:\tools\protobuf\include' `
    -PerlPath 'C:\Strawberry\perl\bin\perl.exe' `
    -InstallRoot 'C:\bk\x64'
```

Example **ARM64** invocation:

```powershell
.\buildkite\windows\provision-pool.ps1 `
    -Architecture arm64 `
    -PythonExecutable 'C:\Python312-arm64\python.exe' `
    -CudaPath 'C:\CUDA\v13.4' `
    -VisualStudioPath 'C:\Program Files\Microsoft Visual Studio\18\BuildTools' `
    -RustVisualStudioPath 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools' `
    -RequirementsFile 'C:\pool-inputs\requirements-arm64.txt' `
    -Wheelhouse 'C:\pool-inputs\wheels-arm64' -NoIndex `
    -CudaArchList '12.0+PTX;10.3a' `
    -CMakeCudaArchitectures '120-real;103-real' `
    -ProtocPath 'C:\tools\protobuf\bin\protoc.exe' `
    -ProtocIncludePath 'C:\tools\protobuf\include' `
    -PerlPath 'C:\tools\perl\bin\perl.exe' `
    -InstallRoot 'C:\bk\arm64'
```

Git, Cargo and Rust must already be on this account's PATH. Rust must be 1.95+
and use the native MSVC host target. `VisualStudioPath` and
`RustVisualStudioPath` may also point directly to `vcvarsall.bat`.
Use `-Queue` for a custom queue and `-MaxJobs` to tune compilation concurrency.
The ARM64 toolset/SDK defaults match the fork helper; override
`-MsvcToolsetVersion`, `-RustMsvcToolsetVersion`, or `-WindowsSdkVersion` if
your tested configuration differs. Buildkite defaults to release `4.3.0`;
`-AgentVersion` selects another explicit stable release, never a moving `latest`.

On success, the root contains `venv`, `bin\buildkite-agent.exe`,
`buildkite-agent.cfg`, `environment.json`, `provisioning.json`,
`start-agent.ps1`, `checkouts`, and `work`. The launcher sets the per-agent
environment shown above, so you do not need to set it manually. It preserves
the agent's exit code and restores the calling process environment on exit.

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

Provisioning never overwrites an existing root or edits a running pool. For an
upgrade, use a new root, then switch the agent launcher after draining the old
agent. A failure leaves its partial directory for diagnosis and does not write
the `provisioning.json` completion marker; the launcher refuses incomplete
installations. A successful provision establishes dependency/tool availability,
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
python -m pytest buildkite\tests\test_windows_ci.py buildkite\tests\test_windows_provisioning.py
```

Native `.cmd` exit-code tests run on Windows and are skipped on other hosts.
Provisioning tests require PowerShell 7; they mock tool installation and agent
startup and never register an agent or install CUDA/toolchains.
Actual wheel compilation and CUDA kernel execution require the provisioned
private pools; infrastructure tests do not establish GPU compatibility.
