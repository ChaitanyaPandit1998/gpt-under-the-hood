#!/usr/bin/env python3
"""
Quick PyTorch/CUDA environment check for this project.

Run with:
  .venv\Scripts\python.exe gpu_check.py
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"Python executable: {sys.executable}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        print("No CUDA device visible to PyTorch.")
        return 1

    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")

    for idx in range(device_count):
        props = torch.cuda.get_device_properties(idx)
        print(
            f"GPU {idx}: {torch.cuda.get_device_name(idx)} | "
            f"compute capability {props.major}.{props.minor} | "
            f"VRAM {props.total_memory / (1024 ** 3):.2f} GiB"
        )

    test = torch.randn(1024, 1024, device="cuda")
    print(f"Test tensor device: {test.device}")
    print("GPU check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
