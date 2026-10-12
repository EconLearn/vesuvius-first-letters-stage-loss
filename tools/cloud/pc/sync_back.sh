#!/bin/bash
# pod -> workstation: ONLY small outputs (PNG strips/crops, JSON/CSV scores, text logs). Never CT, renders, maps or npy.
#   POD=... PORT=... [KEY=...] POD_ROOT_REMOTE=... bash tools/cloud/pc/sync_back.sh [--dry-run] [--with-ckpts] [--with-flat]
# --dry-run lists what would come and the total size first (check the free space of the destination before pulling).
# --with-ckpts adds the fitted checkpoints (< 1 GB each); --with-flat the flattened tifxyz (~100 MB per scroll).
# Destination: $SYNC_DST (default $VC_WORK/cloud/results_pod), same relative paths as under $POD_ROOT_REMOTE on the pod.
# Remote sub-directories follow the defaults of jobs/lib.sh; override with POD_RESULTS_SUB (default out/cloud, =
# RESULTS_DIR relative to the pod's VC_ROOT), POD_QUIET_SUB (work/quiet = QUIET_WORK), POD_READ_SUB (work/read = READ_WORK),
# POD_RUNS_SUB (work/quiet/runs = QUIET_RUNS), POD_STAGE2_SUB (work/ensemble/stage2x), POD_TIMING_SUB (work/cloud/timing).
set -uo pipefail
. "$(dirname "$0")/pod_env.sh"
DST=${SYNC_DST:-${VC_WORK:-${VC_ROOT:-$HOME/vesuvius}/work}/cloud/results_pod}
DRYF=""; CK=0; FL=0
for a in "$@"; do case $a in --dry-run) DRYF=--dry-run;; --with-ckpts) CK=1;; --with-flat) FL=1;; esac; done
mkdir -p "$DST"
R=$POD_ROOT_REMOTE
S_RES=${POD_RESULTS_SUB:-out/cloud}; S_Q=${POD_QUIET_SUB:-work/quiet}; S_R=${POD_READ_SUB:-work/read}
S_RUNS=${POD_RUNS_SUB:-work/quiet/runs}; S_S2=${POD_STAGE2_SUB:-work/ensemble/stage2x}; S_T=${POD_TIMING_SUB:-work/cloud/timing}
SMALL=(--exclude='*.zarr/***' --include='*/' --include='*.png' --include='*.json' --include='*.csv' --include='*.tsv' --include='*.txt'
       --include='*.log' --include='*.super' --include='*.md' --exclude='*')
pull() {  # REMOTE_SUBDIR [extra rsync args]
  local sub=$1; shift
  rsh "test -d $R/$sub" || { echo "  $sub: (absent on the pod)"; return 0; }
  rsync -a $DRYF --mkpath --stats --prune-empty-dirs --max-size=60m ${RSYNC_E[@]+"${RSYNC_E[@]}"} "$@" "$(RPATH "$R/$sub/")" "$DST/$sub/" 2>&1 \
     | grep -E '^(Number of regular files transferred|Total transferred file size)' | sed "s|^|  $sub: |"
}
echo "sync_back from $POD to $DST $DRYF"
pull cloud_state "${SMALL[@]}"
pull "$S_RES" "${SMALL[@]}"
pull "$S_S2" "${SMALL[@]}"
pull "$S_T" "${SMALL[@]}"
pull "$S_Q/logs" --include='*.log' --include='*.super' --exclude='*'
pull "$S_Q/out" --include='*.png' --include='*.json' --exclude='*'
if [ $CK = 1 ]; then rsync -a $DRYF --mkpath --info=progress2 --prune-empty-dirs ${RSYNC_E[@]+"${RSYNC_E[@]}"} --include='*/' --include='checkpoint_fitted.ckpt' --exclude='*' "$(RPATH "$R/$S_RUNS/")" "$DST/$S_RUNS/"; fi
if [ $FL = 1 ]; then rsync -a $DRYF --mkpath --info=progress2 --prune-empty-dirs ${RSYNC_E[@]+"${RSYNC_E[@]}"} --exclude='*.source.tifxyz/***' --include='*/' --include='**/*.tifxyz/*.tif' --include='**/*.tifxyz/meta.json' --exclude='*' "$(RPATH "$R/$S_R/")" "$DST/$S_R/"; fi
[ -z "$DRYF" ] && du -sh "$DST" | sed 's/^/total on the workstation: /'
exit 0
