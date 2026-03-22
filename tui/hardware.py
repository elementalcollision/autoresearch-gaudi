"""Intel Gaudi 3 hardware info for TUI display."""

import subprocess
import re


def get_hardware_summary() -> dict:
    """Get hardware info without importing ML frameworks.

    Returns dict with: chip_name, gpu_cores, total_memory_gb, peak_tflops, device_count
    """
    info = {
        'chip_name': 'Unknown',
        'gpu_cores': 64,  # TPCs per Gaudi 3 device
        'total_memory_gb': 0,
        'peak_tflops': 0.0,
        'device_count': 0,
    }

    # Get device info via hl-smi
    try:
        result = subprocess.run(
            ['hl-smi', '-Q', 'name,memory.total', '-f', 'csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            info['device_count'] = len(lines)

            # Parse first device: "Gaudi 3, 131072 MiB"
            first = lines[0].strip()
            parts = [p.strip() for p in first.split(',')]
            if len(parts) >= 1:
                info['chip_name'] = parts[0]
            if len(parts) >= 2:
                m = re.search(r'(\d+)', parts[1])
                if m:
                    mem_mib = int(m.group(1))
                    info['total_memory_gb'] = mem_mib / 1024

            # Gaudi 3: ~1835 TFLOPS bf16 per device
            info['peak_tflops'] = 1835.0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        # Fallback defaults for Gaudi 3
        info['chip_name'] = 'Gaudi 3'
        info['device_count'] = 8
        info['total_memory_gb'] = 128
        info['peak_tflops'] = 1835.0

    return info
