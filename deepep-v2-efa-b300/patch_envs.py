"""Patch deep_ep/utils/envs.py to read RDMA NIC speed from sysfs (works on EFA)
before falling back to ibstat (which doesn't work on EFA).

DeepEP V2's get_rdma_gbs uses `ibstat <nic>` to parse `Rate: <Gb/s>`. EFA NICs
show up in `ibv_devices` and in `/sys/class/infiniband/<dev>/ports/1/rate` but
not in `ibstat`, so the upstream code returns 0 -> ZeroDivisionError downstream.

This patch tries sysfs first; on EFA it reads "100 Gb/sec" and returns 12.5.
"""

from pathlib import Path
import re
import sys

target = Path('/usr/local/lib/python3.12/dist-packages/deep_ep/utils/envs.py')
src = target.read_text()

OLD = '''@functools.lru_cache()
def get_rdma_gbs(nic_name: str = _DEFAULT_NIC_NAME) -> float:
    """
    Get the RDMA bandwidth in GB/s, cached.

    Arguments:
        nic_name: the NIC device name.

    Returns:
        gbs: the RDMA bandwidth in GB/s (0 if detection fails).
    """
    # noinspection PyBroadException
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout

        pattern = rf"CA '{nic_name}'.*?Port \\d+:\\s*.*?Rate:\\s*(\\d+)"
        match = re.search(pattern, output, re.DOTALL)
        assert match
        rate = int(match.group(1))
        return rate / 8
    except Exception as e:
        print(f'Failed to get RDMA connection speed: {e}')
        return 0'''

NEW = '''@functools.lru_cache()
def get_rdma_gbs(nic_name: str = _DEFAULT_NIC_NAME) -> float:
    """
    Get the RDMA bandwidth in GB/s, cached.

    EFA-friendly: tries sysfs first (works on EFA), falls back to ibstat (IB/RoCE).
    Override with EP_RDMA_GBS env var if both fail.
    """
    # 0. Explicit override.
    env_override = os.getenv('EP_RDMA_GBS')
    if env_override:
        try:
            return float(env_override)
        except ValueError:
            pass

    # 1. sysfs path: /sys/class/infiniband/<nic>/ports/1/rate -> "100 Gb/sec (...)"
    try:
        rate_file = f'/sys/class/infiniband/{nic_name}/ports/1/rate'
        with open(rate_file) as f:
            m = re.match(r'\\s*(\\d+)\\s*Gb', f.read())
        if m:
            return int(m.group(1)) / 8
    except Exception:
        pass

    # 2. Fallback to ibstat (works on Mellanox CX-X / IB / RoCE).
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, check=True)
        output = result.stdout
        pattern = rf"CA '{nic_name}'.*?Port \\d+:\\s*.*?Rate:\\s*(\\d+)"
        match = re.search(pattern, output, re.DOTALL)
        assert match
        return int(match.group(1)) / 8
    except Exception as e:
        print(f'Failed to get RDMA connection speed: {e}')
        return 0'''

if OLD not in src:
    print('ERROR: get_rdma_gbs() did not match expected source. '
          'Has the upstream changed? Aborting.', file=sys.stderr)
    sys.exit(1)

target.write_text(src.replace(OLD, NEW))
print(f'Patched {target}')
