#!/bin/bash
# workstation -> pod upload of fit inputs that already exist locally (instead of rebuilding them on the pod).
#   POD=... PORT=... [KEY=...] POD_ROOT_REMOTE=... bash tools/cloud/pc/upload_inputs.sh --test-only    # uplink speed test only (100 MB)
#   POD=... PORT=... [KEY=...] POD_ROOT_REMOTE=... bash tools/cloud/pc/upload_inputs.sh [--with-ckpt TAG]...
# Uploads, for the datasets in $UPLOAD_SETS (default: the rows of screen_scrolls.tsv with upload=1) that are complete
# locally (normals with BUILD_SUMMARY + tracks + crossings): lasagna_inputs (WITHOUT the *.respool* sidecars, which the
# fit rebuilds) and tracks/; optionally the finished checkpoint of fit TAG (--with-ckpt TAG, repeatable; job 3 then skips
# that fit). Datasets that are not in the set are built on the pod.
# All .upload_in_progress markers are written FIRST so pod jobs wait instead of starting a duplicate build; each
# dataset gets .upload_done + .trk_done when its rsync completes. rsync --partial: re-run to resume.
set -uo pipefail
. "$(dirname "$0")/pod_env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
SETS=${UPLOAD_SETS:-$(awk -F'\t' '$1 !~ /^#/ && $1 != "dataset" && $6 == 1 {printf "%s ", $1}' "${SCREEN_CONFIG:-$HERE/../screen_scrolls.tsv}")}
WITH_CK=(); TEST_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in --test-only) TEST_ONLY=1;; --with-ckpt) shift; WITH_CK+=("${1:?--with-ckpt needs a fit tag}");; esac
  shift
done
L=${VC_DATA:-${VC_ROOT:-$HOME/vesuvius}/data}/spiral          # local fit inputs
R=$POD_DATA_REMOTE/spiral
speed_test() {
  local t0 t1 gb=0 s
  for s in $SETS; do
    [ -d "$L/$s/lasagna_inputs" ] && gb=$(python3 -c "print($gb + $(du -sb --exclude='*respool*' "$L/$s/lasagna_inputs" "$L/$s/tracks" 2>/dev/null | awk '{t+=$1} END {print t+0}') / 1e9)")
  done
  t0=$(date +%s.%N)
  if [ "$POD" = local ]; then head -c 100000000 /dev/urandom > /dev/null
  else head -c 100000000 /dev/urandom | ssh "${SSHOPT[@]}" "$POD" 'cat > /dev/null'; fi
  t1=$(date +%s.%N)
  python3 - "$t0" "$t1" "$gb" <<'PYEOF'
import sys
dt = float(sys.argv[2]) - float(sys.argv[1]); mbit = 800 / dt; gb = float(sys.argv[3])
print(f'uplink ~{mbit:.0f} Mbit/s (100 MB in {dt:.1f} s); {gb:.1f} GB of local inputs -> ~{gb * 1e3 * 8 / mbit / 60:.0f} min')
print('DECISION RULE (RUNBOOK): >= 20 Mbit/s -> upload (INPUT_MODE=upload); below -> rebuild on the pod (INPUT_MODE=rebuild)')
PYEOF
}
speed_test
[ "$TEST_ONLY" = 1 ] && exit 0
for S in $SETS; do
  [ -f "$L/$S/lasagna_inputs/BUILD_SUMMARY.json" ] && ls "$L/$S"/tracks/*.dbm.crossings.npz > /dev/null 2>&1 || { echo "$S is not complete locally; skipping"; continue; }
  rsh "mkdir -p $R/$S && touch $R/$S/.upload_in_progress"
done
for S in $SETS; do
  [ -f "$L/$S/lasagna_inputs/BUILD_SUMMARY.json" ] || continue
  echo "== $S $(du -sh --exclude='*respool*' "$L/$S/lasagna_inputs" "$L/$S/tracks" | awk '{print $1}' | tr '\n' ' ')"
  if rsync -a --partial --info=progress2 ${RSYNC_E[@]+"${RSYNC_E[@]}"} --exclude '*respool*' "$L/$S/lasagna_inputs" "$L/$S/tracks" "$(RPATH "$R/$S/")"; then
    rsh "cd $R/$S && date -Is > .upload_done && date -Is > .trk_done && rm -f .upload_in_progress && echo '$S upload done'"
  else
    echo "$S rsync failed; re-run this script to resume (the pod waits up to UPLOAD_WAIT_H, then builds)"
  fi
done
for TAG in ${WITH_CK[@]+"${WITH_CK[@]}"}; do
  CK=${QUIET_RUNS:-${VC_WORK:-${VC_ROOT:-$HOME/vesuvius}/work}/quiet/runs}/$TAG/checkpoint_fitted.ckpt
  RUNS_REMOTE=${POD_RUNS_REMOTE:-$POD_ROOT_REMOTE/work/quiet/runs}
  rsh "mkdir -p $RUNS_REMOTE/$TAG"
  rsync -a --partial --info=progress2 ${RSYNC_E[@]+"${RSYNC_E[@]}"} "$CK" "$(RPATH "$RUNS_REMOTE/$TAG/")" && echo "$TAG checkpoint uploaded"
done
