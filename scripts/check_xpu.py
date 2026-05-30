"""快速验证 Intel Arc XPU 是否可用。运行: .venv-xpu\Scripts\python scripts\check_xpu.py"""
import torch

print("torch:", torch.__version__)
print("xpu available:", torch.xpu.is_available())
if torch.xpu.is_available():
    print("device count:", torch.xpu.device_count())
    print("device name:", torch.xpu.get_device_name(0))
    # 实际算一把，确认 kernel 能跑
    a = torch.randn(2048, 2048, device="xpu")
    b = torch.randn(2048, 2048, device="xpu")
    c = a @ b
    torch.xpu.synchronize()
    print("matmul on xpu ok, result sum:", c.sum().item())
else:
    print("XPU NOT available — 检查 GPU 驱动 / oneAPI runtime")
