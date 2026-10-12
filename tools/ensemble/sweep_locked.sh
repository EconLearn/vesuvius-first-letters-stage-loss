#!/bin/bash
# GPU wrapper for depth_sweep.py: runs it under a GPU-lock wrapper script (GPU_LOCKED, default
# tools/cloud/gpu/gpu_locked.sh; called as `gpu_locked.sh LABEL command...`). The contract the wrapper satisfies (and any
# replacement must): take a GPU lock file, wait until no fit_spiral.py runs and no resumable GPU job is queued, wait for
# the card to drain (<= 1500 MiB used), run the command, release the lock.
# depth_sweep.py itself refuses --device cuda unless ENS_GPU_LOCK_HELD=1 (set only here) and re-checks that no
# fit_spiral.py runs.
# usage: sweep_locked.sh LABEL [depth_sweep.py args...]
#   env: VC_ROOT (default ~/vesuvius), PY (python, default $VC_ROOT/.venv/bin/python), GPU_LOCKED (wrapper path)
LABEL=${1:?label}; shift
VC_ROOT=${VC_ROOT:-$HOME/vesuvius}                       # see tools/vcpaths.py
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # tools/ensemble
PY=${PY:-$VC_ROOT/.venv/bin/python}
GPU_LOCKED=${GPU_LOCKED:-$HERE/../cloud/gpu/gpu_locked.sh}
[ -f "$GPU_LOCKED" ] || { echo "sweep_locked.sh: GPU-lock wrapper not found: $GPU_LOCKED (set GPU_LOCKED)" >&2; exit 2; }
if [ -x "$GPU_LOCKED" ]; then WRAP=("$GPU_LOCKED"); else WRAP=(bash "$GPU_LOCKED"); fi    # works without the exec bit
export ENS_GPU_LOCK_HELD=1
exec "${WRAP[@]}" "ens-$LABEL" "$PY" "$HERE/depth_sweep.py" "$@"
