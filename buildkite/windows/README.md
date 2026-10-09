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
python -m pytest buildkite\tests\test_windows_ci.py
```

Native `.cmd` exit-code tests run on Windows and are skipped on other hosts.
Actual wheel compilation and CUDA kernel execution require the provisioned
private pools; infrastructure tests do not establish GPU compatibility.
