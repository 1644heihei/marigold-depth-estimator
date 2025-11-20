import sys
import os
import torch

print("=" * 70)
print("PyTorch CUDA Diagnostics")
print("=" * 70)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version (torch.version.cuda): {torch.version.cuda}")

# CUDA のビルド情報を確認
try:
    print(f"CUDA arch list: {torch.cuda.get_arch_list()}")
except Exception as e:
    print(f"CUDA arch list error: {e}")

# パス確認
print(f"\nCUDA_HOME env var: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"PATH (first 200 chars): {os.environ.get('PATH', 'Not set')[:200]}")

# 試しに CUDA デバイスにアクセス
print("\nTrying to create tensor on CUDA device...")
try:
    x = torch.zeros(1, device="cuda")
    print("✓ Successfully created tensor on CUDA device")
except Exception as e:
    print(f"✗ CUDA tensor creation error: {e}")

# GPU 情報
if torch.cuda.is_available():
    print(f"\n✓ GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("\n✗ CUDA not available - GPU support disabled")

print("=" * 70)
