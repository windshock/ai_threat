"""
Repo entrypoint wrapper
======================
This repository's actual demo lives under:
  inference-runtime/pytorch-inference-time-latent-steering/activation_hook_demo.py

The root README references `python activation_hook_demo.py`, so this thin wrapper
delegates execution to the real script while preserving CLI/env behavior.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = (
        Path(__file__).resolve().parent
        / "inference-runtime"
        / "pytorch-inference-time-latent-steering"
        / "activation_hook_demo.py"
    )
    if not target.exists():
        raise SystemExit(f"Expected demo script not found: {target}")

    # Make stack traces / help messages point at the real script.
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()


