#!/usr/bin/env bash
# Upgrade a p6-b300 HOST from EFA installer 1.49.0 to 1.50.0 and prove that
# NCCL_GIN_TYPE=5 (EFA GDA) can actually run afterwards.
#
# Run this ON the host:  bash prepare_host_efa150.sh
#
# Why the host and not the container: the container image already carries EFA
# userspace 1.50.0 (rdma-core 64.0, libfabric 2.6.0), but the COMP_CNTR capability
# that GIN type 5 needs is a property of efa.ko. On this host efa.ko is 3.1.0g and
# every one of the 16 EFA devices fails ibv_create_comp_cntr with errno 93
# (EOPNOTSUPP). Only the 1.50.0 installer's kmod (efa 3.3.0) exposes it, so the
# upgrade cannot be done from inside a container.
#
# It REQUIRES IDLE GPUs AND IDLE EFA: the installer unloads efa.ko, which fails
# while any container holds /dev/infiniband, and a partially reloaded module takes
# the fabric down under whatever is running. The preflight refuses in that case;
# FORCE=1 overrides it (do not).
#
# Env:
#   FORCE=1        skip the busy-GPU / busy-EFA preflight
#   PROBE_IMAGE    image used for the post-upgrade capability probe. Default: the
#                  first local deepep-v2-efa-official:sm103-* image. Any image with
#                  rdma-core >= 64.0 headers and gcc works; the probe is compiled
#                  inside it because the HOST's rdma-core is what we just replaced.
#   SKIP_PROBE=1   upgrade only, no probe
set -euo pipefail

TARBALL="${TARBALL:-$HOME/work/ep-benchmarks-efa/deepep-v2-efa-official/aws-efa-installer-1.50.0.tar.gz}"
PROBE_SRC="${PROBE_SRC:-$HOME/work/ep-benchmarks-efa/deepep-v2-efa-official/ce_probe.c}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$HOME/efa150_upgrade_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== log: $LOG"

step () { printf '\n########## %s\n' "$1"; }

step "0. preflight -- this host must be idle"
[ -f "$TARBALL" ] || { echo "!! missing $TARBALL"; exit 1; }
sz=$(stat -c %s "$TARBALL")
[ "$sz" -gt 600000000 ] || { echo "!! $TARBALL is only $sz bytes, expected ~650 MB"; exit 1; }
echo "tarball $TARBALL ($sz bytes)"

busy=0
apps=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true)
if [ -n "$apps" ]; then echo "!! GPUs are in use:"; echo "$apps"; busy=1; else echo "GPUs: no compute processes"; fi
# A container can hold the EFA devices without showing a compute process.
ibusers=$(docker ps -q 2>/dev/null | while read -r c; do
            docker inspect -f '{{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$c" 2>/dev/null
          done | grep -E 'infiniband' || true)
if [ -n "$ibusers" ]; then echo "!! containers holding /dev/infiniband:"; echo "$ibusers"; busy=1; else echo "EFA: no container holds /dev/infiniband"; fi
# fuser on the char devices catches a bare-metal holder too.
holders=$(sudo fuser -v /dev/infiniband/uverbs* 2>&1 | tail -n +2 || true)
[ -z "$holders" ] || { echo "!! /dev/infiniband/uverbs* held by:"; echo "$holders"; busy=1; }
if [ "$busy" = 1 ] && [ "${FORCE:-}" != 1 ]; then
  echo
  echo "REFUSING: the installer unloads efa.ko. Stop the tenants first"
  echo "  docker ps        # then docker stop <them>"
  echo "and re-run. FORCE=1 overrides (it will break the running job)."
  exit 3
fi

step "1. before-state"
kver=$(uname -r); echo "kernel        : $kver"
echo "efa kmod      : $(modinfo efa 2>/dev/null | awk '/^version:/{print $2}')"
grep -m1 'EFA installer version' /opt/amazon/efa_installed_packages 2>/dev/null || echo "installer     : unknown"
dpkg -l 2>/dev/null | awk '/rdma-core|libfabric1-aws|libnccl-ofi/{print "  " $2 " " $3}'
efa_n=$(ls -d /sys/class/infiniband/rdmap* 2>/dev/null | wc -l)
echo "efa devices   : $efa_n"

step "2. install EFA 1.50.0 (kmod + rdma-core + libfabric + plugin)"
work=$(mktemp -d /tmp/efa150.XXXXXX)
tar -xzf "$TARBALL" -C "$work"
cd "$work/aws-efa-installer"
grep -m1 '## \[1.50.0\]' ChangeLog.md >/dev/null || { echo "!! ChangeLog does not say 1.50.0"; exit 1; }
# Full install, not --minimal: efa-config / efa-profile / limits come with it, and
# leaving 1.49's libfabric behind next to a 3.3.0 kmod is the mixed state that
# produces fi_getinfo failures that look like hardware faults.
sudo ./efa_installer.sh -y

step "3. after-state -- the kmod is the whole point"
newver=$(modinfo efa 2>/dev/null | awk '/^version:/{print $2}')
echo "efa kmod      : $newver"
grep -m1 'EFA installer version' /opt/amazon/efa_installed_packages || true
dpkg -l | awk '/rdma-core|libfabric1-aws|libnccl-ofi/{print "  " $2 " " $3}'
case "$newver" in
  3.3.*) echo "OK: efa.ko $newver is loaded" ;;
  *)     echo
         echo "!! efa.ko is still $newver -- the new module was installed on disk but"
         echo "   the running one could not be replaced (a holder kept it in use)."
         echo "   REBOOT this host, then re-run this script; it is idempotent and will"
         echo "   go straight to the checks."
         exit 4 ;;
esac

step "4. GPUDirect prerequisites (a module reload resets both)"
lsmod | grep -q '^gdrdrv' || sudo modprobe gdrdrv || true
lsmod | grep -E '^gdrdrv|^nvidia_peermem|^efa ' || true
if [ ! -e /dev/gdrdrv ]; then
  maj=$(awk '/gdrdrv/{print $1}' /proc/devices | head -1)
  if [ -n "$maj" ]; then
    echo "creating /dev/gdrdrv (major $maj) -- gdrdrv ships no udev rule"
    sudo mknod /dev/gdrdrv c "$maj" 0 && sudo chmod 666 /dev/gdrdrv
  else
    echo "!! gdrdrv not in /proc/devices -- DKMS module missing for kernel $kver:"
    dkms status 2>/dev/null | grep -i gdrdrv || echo "   (no gdrdrv in dkms status)"
  fi
fi
ls -l /dev/gdrdrv 2>/dev/null || echo "!! /dev/gdrdrv absent -- DeepEP will fail GDRCopy init"
systemctl is-active nvidia-fabricmanager >/dev/null 2>&1 \
  && echo "fabricmanager : active" \
  || echo "!! fabricmanager not active -- NCCL will hang silently on the first collective"
nvidia-smi -q 2>/dev/null | grep -A2 -i 'Fabric' | head -8 || true

step "5. capability probe -- 16x CE OK is the gate for NCCL_GIN_TYPE=5"
if [ "${SKIP_PROBE:-}" = 1 ]; then echo "skipped (SKIP_PROBE=1)"; exit 0; fi
img="${PROBE_IMAGE:-$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -m1 '^deepep-v2-efa-official:sm103')}"
[ -n "$img" ] || { echo "!! no probe image; set PROBE_IMAGE=<image with rdma-core >= 64.0>"; exit 1; }
echo "probe image   : $img"
[ -f "$PROBE_SRC" ] || { echo "!! missing $PROBE_SRC"; exit 1; }
# Throwaway container, never a live one: it opens every device.
docker run --rm --privileged --network=host \
  --device=/dev/infiniband \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v "$PROBE_SRC":/tmp/ce_probe.c:ro \
  "$img" bash -lc 'gcc -O0 -o /tmp/ce_probe /tmp/ce_probe.c -libverbs && /tmp/ce_probe' \
  | tee /tmp/ce_probe_after.txt
okc=$(grep -c 'CE OK' /tmp/ce_probe_after.txt || true)
echo
echo "=== CE OK on $okc device(s)"
if [ "$okc" -ge 16 ]; then
  echo "PASS: EFA GDA (NCCL_GIN_TYPE=5) is available on all 16 EFA NICs."
else
  echo "FAIL: expected 16. errno 93 = the kmod still lacks COMP_CNTR (reboot?),"
  echo "      errno 95 = an IB device, which is expected for the 2 CX-7 ports."
  grep 'CE FAIL' /tmp/ce_probe_after.txt | head -20
  exit 5
fi
