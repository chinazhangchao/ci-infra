"""Pin vLLM imports to the wheel in the disposable test venv."""

import sys
from importlib.machinery import PathFinder
from pathlib import Path


class WheelPackageFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "vllm":
            return None
        spec = PathFinder.find_spec(fullname, [str(Path(__file__).resolve().parent)])
        if spec is None:
            raise ModuleNotFoundError(
                "The test venv has no installed vLLM wheel; refusing source fallback.",
                name=fullname,
            )
        return spec


# Installed as a .pth startup hook only inside the job's disposable test venv.
sys.meta_path.insert(0, WheelPackageFinder())
