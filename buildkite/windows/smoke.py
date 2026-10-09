"""Exercise the installed wheel, not the source checkout or a cached vLLM."""

import argparse
import importlib
import json
import sys
import sysconfig
from pathlib import Path


def smoke(package_dir, architecture):
    expected = {"x64": "win-amd64", "arm64": "win-arm64"}[architecture]
    if sys.platform != "win32" or sysconfig.get_platform() != expected:
        raise RuntimeError(f"Expected native {expected} Python.")
    sys.path.insert(0, str(package_dir))
    import torch
    import vllm

    if not Path(vllm.__file__).resolve().is_relative_to(package_dir):
        raise RuntimeError(
            "Smoke test imported vLLM outside the freshly installed wheel."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; the smoke test cannot be skipped.")

    importlib.import_module("vllm._custom_ops")
    extension = importlib.import_module("vllm._C")
    if not Path(extension.__file__).resolve().is_relative_to(package_dir):
        raise RuntimeError("Smoke test loaded a cached vLLM CUDA extension.")
    torch.manual_seed(0)
    with torch.inference_mode():
        x = torch.randn(7, 1024, dtype=torch.float32, device="cuda")
        output = torch.empty(7, 512, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(output, x)
        torch.cuda.synchronize()
        reference = torch.nn.functional.silu(x[:, :512]) * x[:, 512:]
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)
    return {
        "architecture": architecture,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "kernel": "_C.silu_and_mul",
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--architecture", choices=["x64", "arm64"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = smoke(args.package_dir.resolve(), args.architecture)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
