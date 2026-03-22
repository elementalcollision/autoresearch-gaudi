"""
Hardware auto-detection, chip tier classification, and hyperparameter suggestions
for Intel Gaudi 3 accelerators on IBM Cloud.
"""

import os
import subprocess
import re


def detect_backend():
    """
    Verify Gaudi HPU is available.
    Returns 'hpu' or raises RuntimeError.
    """
    try:
        import torch
        if hasattr(torch, 'hpu') and torch.hpu.is_available():
            return "hpu"
    except ImportError:
        raise RuntimeError(
            "PyTorch not found. Ensure you are running inside the Habana Docker container "
            "with PyTorch pre-installed."
        )

    raise RuntimeError(
        "Gaudi HPU not available. Ensure habanalabs drivers are installed and "
        "you have sourced /etc/profile.d/habanalabs*.sh"
    )


def get_hardware_info():
    """
    Returns hardware info dict with keys:
      memory_gb, chip_name, chip_tier, gpu_cores, device_count
    Uses hl-smi (Habana System Management Interface) for hardware detection.
    """
    info = {
        "memory_gb": 0,
        "chip_name": "unknown",
        "chip_tier": "gaudi3",
        "gpu_cores": 0,
        "device_count": 0,
    }

    # Try hl-smi for device info
    try:
        result = subprocess.run(
            ["hl-smi", "-Q", "name,memory.total", "-f", "csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            info["device_count"] = len(lines)
            # Parse first line: "Gaudi 3, 131072 MiB"
            first = lines[0].strip()
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 1:
                info["chip_name"] = parts[0]
            if len(parts) >= 2:
                mem_str = parts[1]
                m = re.search(r"(\d+)", mem_str)
                if m:
                    mem_mib = int(m.group(1))
                    info["memory_gb"] = mem_mib / 1024  # MiB to GB
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: try torch.hpu for device count
    if info["device_count"] == 0:
        try:
            import torch
            if hasattr(torch, 'hpu'):
                info["device_count"] = torch.hpu.device_count()
                info["chip_name"] = "Gaudi 3"
                info["memory_gb"] = 128  # Gaudi 3 default: 128 GB HBM2e
        except Exception:
            pass

    # Gaudi 3 doesn't have "GPU cores" in the traditional sense,
    # but we use 2 matrix math engines (MMEs) and 64 tensor processor cores (TPCs)
    info["gpu_cores"] = 64  # TPCs per device

    return info


def get_peak_flops(hw_info=None):
    """
    Peak bf16 FLOPS for MFU calculation.
    Gaudi 3: ~1835 TFLOPS bf16 per accelerator (matrix multiply).
    Returns FLOPS (not TFLOPS) for direct use in MFU computation.
    """
    if hw_info is None:
        hw_info = get_hardware_info()

    # Gaudi 3: 1835 TFLOPS bf16 per card
    flops_per_device = 1835e12
    return flops_per_device


def suggest_hyperparameters(hw_info=None):
    """
    Suggest hyperparameters based on Gaudi 3 hardware.
    Returns dict with: depth, device_batch_size, total_batch_size, eval_tokens_multiplier
    """
    if hw_info is None:
        hw_info = get_hardware_info()

    mem_gb = hw_info.get("memory_gb", 128)

    # Gaudi 3 with 128 GB HBM2e per device — dramatically more capacity
    # than Apple Silicon. Scale model depth and batch sizes accordingly.
    if mem_gb >= 96:
        return {
            "depth": 16,
            "device_batch_size": 64,
            "total_batch_size": 2**18,  # 256K tokens
            "eval_tokens_multiplier": 10,
        }
    elif mem_gb >= 48:
        return {
            "depth": 12,
            "device_batch_size": 32,
            "total_batch_size": 2**17,  # 128K tokens
            "eval_tokens_multiplier": 10,
        }
    else:
        # Unexpected small memory — conservative
        return {
            "depth": 8,
            "device_batch_size": 16,
            "total_batch_size": 2**15,  # 32K tokens
            "eval_tokens_multiplier": 5,
        }


def sync_device(device_type):
    """Synchronize device for accurate timing."""
    if device_type == "hpu":
        import torch
        torch.hpu.synchronize()
    elif device_type == "cuda":
        import torch
        torch.cuda.synchronize()


def get_peak_memory_mb(device_type):
    """Get peak memory usage in MB."""
    if device_type == "hpu":
        import torch
        try:
            return torch.hpu.max_memory_allocated() / 1024 / 1024
        except (AttributeError, RuntimeError):
            return 0.0
    elif device_type == "cuda":
        import torch
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def print_hardware_summary():
    """Print a summary of detected hardware and suggested config."""
    hw = get_hardware_info()
    hp = suggest_hyperparameters(hw)
    peak_flops = get_peak_flops(hw)

    print(f"Hardware: {hw['chip_name']}")
    print(f"  Memory: {hw['memory_gb']:.0f} GB HBM2e")
    print(f"  Devices: {hw['device_count']}")
    print(f"  TPCs per device: {hw['gpu_cores']}")
    print(f"  Peak bf16 FLOPS: {peak_flops:.2e}")
    print(f"Suggested config:")
    print(f"  Depth: {hp['depth']}")
    print(f"  Device batch size: {hp['device_batch_size']}")
    print(f"  Total batch size: {hp['total_batch_size']:,}")


if __name__ == "__main__":
    print_hardware_summary()
    print(f"\nDetected backend: {detect_backend()}")
