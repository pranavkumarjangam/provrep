"""Print whether PyTorch can see the GPU (used by setup_gpu.bat)."""
import torch

ok = torch.cuda.is_available()
print("PyTorch", torch.__version__, "| CUDA available:", ok)
if ok:
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 1024**3:.1f} GB")
    x = torch.randn(1024, 1024, device="cuda")
    print("GPU matmul OK:", float((x @ x).abs().mean()) > 0)
else:
    print("PyTorch cannot see the GPU - check the NVIDIA driver, then re-run setup_gpu.bat")
