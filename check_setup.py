"""Run this first:  python check_setup.py   -> tells you whether PyTorch can use your GPU."""

import sys

print("Python :", sys.version.split()[0])
try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed. Follow step 3 in README.md.")

print("PyTorch:", torch.__version__)
if not torch.cuda.is_available():
    sys.exit("PROBLEM: PyTorch cannot see the GPU. You probably have the CPU-only build.\n"
             "Fix: pip uninstall -y torch   then install again with the command in README.md section 3.")

print("GPU    :", torch.cuda.get_device_name(0))
print("VRAM   :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
try:
    x = torch.randn(2000, 2000, device="cuda")
    (x @ x).sum().item()
except RuntimeError as err:
    sys.exit(f"PROBLEM: the GPU was found but cannot run this PyTorch build:\n{err}\n"
             "RTX 50-series cards need a PyTorch build made for CUDA 12.8 or newer (README.md section 3).")
print("OK: the GPU works. You can start training.")
