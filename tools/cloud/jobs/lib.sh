#!/bin/bash
# Shared helpers for the pod jobs (sourced). Paths follow tools/vcpaths.py: everything lives under $VC_ROOT (default
# ~/vesuvius: data/, work/, ckpt/, ext/, repos/, the venvs, the job state) and the code comes from this repository
# checkout (tools/). Every job logs to ONE status file ($STATE/STATUS.log); detailed logs per step in $STATE/logs/. Steps
# are resumable (done markers in $STATE/done; the underlying tools resume inside a step). Nothing here deletes results;
# the only pruning is of regenerable intermediates (strip renders, CT cache of a finished scroll).
# DRY_RUN=1: heavy tools are replaced by tests/mock_tools.py (local dry run); the control flow is the same.
#
# Locations (env var -> default):
#   VC_ROOT -> ~/vesuvius   VC_DATA -> $VC_ROOT/data   VC_WORK -> $VC_ROOT/work   VC_OUT -> $VC_ROOT/out
#   VC_CKPT -> $VC_ROOT/ckpt   VC_EXT -> $VC_ROOT/ext   POD_STATE -> $VC_ROOT/cloud_state
#   ENS_LANE -> tools/ensemble (render66, depth_sweep, stage2_q343, scorers)
#   SPIRAL_FIT_DIR -> $VILLA_DIR/spiral-fitting (villa checkout made by bootstrap.sh; VILLA_DIR -> $VC_ROOT/repos/villa)
#   REL -> $PHERC343_REPO -> $VC_EXT/pherc343-first-letters   (the released PHerc0343 surfaces, read as data)
#   CALIB343_REGIONS -> $VC_WORK/ensemble/calib343/regions.json (made by tools/ensemble/calib343_regions.py)
#   RESULTS_DIR -> $VC_OUT/cloud   (per-scroll result folders)
#   QUIET_DIR -> tools/cloud/quiet   fit supervisor and input builders (quiet_super.py, build_normals_band.py, tracks_subset.py,
#                                 fetch_lasagna_band.py, umb_auto.py, umb_smooth.py, trk_loadcheck.py, m7normals.py)
#   QUIET_WORK -> $VC_WORK/quiet  their working files: logs/, out/, tmp/, cache/, runs/ (QUIET_RUNS -> $QUIET_WORK/runs, where
#                                 the supervisor writes <tag>/checkpoint_fitted.ckpt), GPU_LOCK
#   READ_DIR -> tools/cloud/read  read_flatten.py (official flatten wrapper); READ_WORK -> $VC_WORK/read holds the flattened
#                                 surfaces <tag>/flat/<tag>.tifxyz
#   GPU_LOCKED -> tools/cloud/gpu/gpu_locked.sh   GPU lock wrapper used by tools/ensemble/sweep_locked.sh (CUDA path of
#                                 stage2_q343.py --run in job 2); it also needs gpu/pending_resume.py
#   The scrolls, z-bands and fit tags screened by the jobs: screen_scrolls.tsv ($SCREEN_CONFIG).
#   SENSE_MODE -> both   for a screened band whose config z_top_to_bottom is '-' (winding sense unknown): both = fit each
#                        sense (<tag>_cw, <tag>_acw) and pick one with quiet/pick_sense.py; cw | acw = fit and take that
#                        sense; fail = do not fit such bands (job 3 skips them until a person picks the sense).
#   SENSE_WINDOW (500), SENSE_MIN_REL_DIFF (0.02): the pick_sense.py rule (see RUNBOOK.md).
#   SENSE_VALIDATE (0) -> 1: job 3 first checks that rule on a band of known sense (SENSE_VALIDATE_SETS, default PHerc0343x)
#                        and stops before any unknown-sense scroll if it does not choose the catalogue sense.
set -uo pipefail
CLOUD=${CLOUD:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # tools/cloud
TOOLS=$(cd "$CLOUD/.." && pwd)                                      # tools/ (vcpaths.py, g12lib.py, ensemble/)
ROOT=${VC_ROOT:-$HOME/vesuvius}
export VC_ROOT=$ROOT
export VC_DATA=${VC_DATA:-$ROOT/data} VC_WORK=${VC_WORK:-$ROOT/work} VC_OUT=${VC_OUT:-$ROOT/out}
export VC_CKPT=${VC_CKPT:-$ROOT/ckpt} VC_EXT=${VC_EXT:-$ROOT/ext}
DATA=$VC_DATA; WORK=$VC_WORK; OUT=$VC_OUT; EXT=$VC_EXT
export POD_STATE=${POD_STATE:-$ROOT/cloud_state}
STATE=$POD_STATE
mkdir -p "$STATE/done" "$STATE/logs"
# operator overrides (budget, rates, scroll list, knobs): one VAR=value per line
[ -f "$STATE/pod.env" ] && . "$STATE/pod.env"
PY=${POD_PY:-$ROOT/.venv/bin/python}
PYX=${POD_PYX:-$ROOT/.venv-x2/bin/python}
export POD_PY=$PY
LE=${ENS_LANE:-$TOOLS/ensemble}
export ENS_LANE=$LE
export POD_PYX=$PYX
LQ=${QUIET_DIR:-$CLOUD/quiet}; export QUIET_DIR=$LQ
QW=${QUIET_WORK:-$WORK/quiet}; export QUIET_WORK=$QW
QRUNS=${QUIET_RUNS:-$QW/runs}; export QUIET_RUNS=$QRUNS
LR=${READ_DIR:-$CLOUD/read}; export READ_DIR=$LR
RW=${READ_WORK:-$WORK/read}; export READ_WORK=$RW
export GPU_LOCKED=${GPU_LOCKED:-$CLOUD/gpu/gpu_locked.sh}
SF=${SPIRAL_FIT_DIR:-${VILLA_DIR:-$ROOT/repos/villa}/spiral-fitting}; export SPIRAL_FIT_DIR=$SF
DRY=${DRY_RUN:-0}
MOCK="$PY $CLOUD/tests/mock_tools.py"
REL=${REL:-${PHERC343_REPO:-$EXT/pherc343-first-letters}}
REGIONS=${CALIB343_REGIONS:-$WORK/ensemble/calib343/regions.json}
RESULTS=${RESULTS_DIR:-$OUT/cloud}
# disk-guard shim for scripts outside this repository (pod/sitecustomize.py): only active when POD_GUARD_PATH is set
if [ -n "${POD_GUARD_PATH:-}" ]; then export POD_GUARD_PATH; fi
if [ -n "${POD_GUARD_OFFSET_GB:-}" ]; then export POD_GUARD_OFFSET_GB; fi
# knobs (defaults; override in $STATE/pod.env)
: "${DEVICE:=cuda}" "${BATCH:=0}" "${RENDER_PROCS:=3}" "${STOP_IT_S:=3.0}" "${TRACK_START:=7000}"
export FIT_SPIRAL_TRITON=${FIT_SPIRAL_TRITON:-1}  # the non-Triton inverse builds points x windings and OOMed the 24 GB 4090 on a 100M-point band; the Triton path used 6 GB
export VILLA_FAST=${VILLA_FAST:-1}  # ensemble/speed/fastvilla via py/sweep.py; bit-identical on the pod (236 M px, 0 differing) and 2.4-2.8x faster on the 4090
: "${GPU_KILL_MB:=23500}" "${GPU_START_MAX_MB:=3000}" "${PRUNE_CT:=1}" "${OVERLAP:=0}" "${INPUT_WORKERS:=$(( $(nproc) > 16 ? 12 : 3 ))}"
: "${RATE_USD_H:=0.34}" "${BUDGET_USD:=25}"
: "${SENSE_MODE:=both}"; export SENSE_MODE
export SENSE_WINDOW SENSE_MIN_REL_DIFF    # optional: the pick_sense.py rule parameters (defaults are in that script)
export ENS_NORM_WORKERS=${ENS_NORM_WORKERS:-6}
JOB=${JOB:-job}

# screening configuration: SC[dataset]=scroll, BAND[dataset]="z0 z1", FJ[dataset]="fit_tag z0 z1" from screen_scrolls.tsv;
# SCREEN_DEFAULT = datasets of group 'screen' (job 3 default $SCROLLS), UPLOAD_DEFAULT = datasets with upload=1
SCREEN_CONFIG=${SCREEN_CONFIG:-$CLOUD/screen_scrolls.tsv}; export SCREEN_CONFIG
declare -A SC BAND FJ ZD VX
SCREEN_DEFAULT=""; UPLOAD_DEFAULT=""
while IFS=$'\t' read -r _ds _sc _tag _z0 _z1 _up _grp _vid _sz _sy _sx _zd _vx _rest; do
  case $_ds in ''|'#'*|dataset) continue ;; esac
  SC[$_ds]=$_sc; BAND[$_ds]="$_z0 $_z1"; FJ[$_ds]="$_tag $_z0 $_z1"; ZD[$_ds]=$_zd
  VX[$_ds]=${_vx:-8.64}; [ "${VX[$_ds]}" = "-" ] && VX[$_ds]=8.64      # voxel size in um
  if [ "$_grp" = screen ]; then SCREEN_DEFAULT="${SCREEN_DEFAULT:+$SCREEN_DEFAULT }$_ds"; fi
  if [ "$_up" = 1 ]; then UPLOAD_DEFAULT="${UPLOAD_DEFAULT:+$UPLOAD_DEFAULT }$_ds"; fi
done < "$SCREEN_CONFIG"
unset _ds _sc _tag _z0 _z1 _up _grp _vid _sz _sy _sx _zd _vx _rest
case $SENSE_MODE in both|cw|acw|fail) ;; *) echo "SENSE_MODE must be both, cw, acw or fail (got $SENSE_MODE)" >&2; exit 2 ;; esac

status() { echo "$(date -Is) [$JOB] $*" | tee -a "$STATE/STATUS.log"; }
# the budget watchdog (or an operator) stops a job with SIGTERM to its process group: log it, exit 3 like a guard stop
trap 'status "terminated by SIGTERM (budget watchdog or operator); results so far are kept"; exit 3' TERM

# the job runs in its own process group so the budget watchdog can stop exactly this job
own_group() {  # call as: own_group "$0" "$@"
  if [ "$(ps -o pgid= $$ | tr -d ' ')" != "$$" ] && [ -z "${POD_REEXEC:-}" ]; then
    POD_REEXEC=1 exec setsid bash "$@"
  fi
}

job_lock() {   # one instance per job, and one GPU job at a time
  exec 8> "$STATE/.$JOB.lock"; flock -n 8 || { status "another $JOB is running; exiting"; exit 9; }
  exec 9> "$STATE/.gpu_jobs.lock"
  flock -n 9 || { status "another GPU job holds $STATE/.gpu_jobs.lock; waiting"; flock 9; }
}

start_watchdog() {
  local pg; pg=$(ps -o pgid= $$ | tr -d ' ')
  # 8>&- 9>&-: background helpers must not inherit (and so hold) the job / GPU locks
  setsid nohup "$PY" "$CLOUD/py/cost_guard.py" watch "$pg" > "$STATE/logs/watchdog_$JOB.log" 2>&1 < /dev/null 8>&- 9>&- &
}

guard() {  # guard LABEL [EST_H] -> 0 go, 3 budget stop, 4 would exceed
  if [ -n "${2:-}" ]; then "$PY" "$CLOUD/py/cost_guard.py" check "$JOB:$1" --est-h "$2"
  else "$PY" "$CLOUD/py/cost_guard.py" check "$JOB:$1"; fi
}

is_done() { [ -e "$STATE/done/${JOB}__$1" ]; }

step() {   # step NAME EST_HOURS CMD... : skip if done; cost guard; run with log; mark done on success
  local name=$1 est=$2; shift 2
  if is_done "$name"; then status "skip $name (done earlier)"; return 0; fi
  guard "$name" "$est"; local g=$?
  if [ $g -ne 0 ]; then status "STOP before $name: cost guard rc=$g"; exit $g; fi
  status "START $name (est ${est} h)"
  local t0; t0=$(date +%s)
  "$@" >> "$STATE/logs/$JOB.$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    date -Is > "$STATE/done/${JOB}__$name"; status "DONE $name in $(( $(date +%s) - t0 )) s"
  else
    status "FAIL $name rc=$rc after $(( $(date +%s) - t0 )) s (log: $STATE/logs/$JOB.$name.log)"
  fi
  return $rc
}

gpu_sampler() {  # gpu_sampler OUT.csv -> pid ; 1 s samples of util + memory
  # stdout/stderr must not stay attached to the caller's $(...) pipe, or the substitution never returns
  if [ "$DRY" = 1 ]; then $MOCK gpu "$1" > /dev/null 2>&1 < /dev/null & echo $!; return; fi
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$1" 2>/dev/null < /dev/null &
  echo $!
}

ckpt_iters() {  # ckpt_iters TAG -> completed iterations or -1
  local ck=$QRUNS/$1/checkpoint_fitted.ckpt
  [ -f "$ck" ] || { echo -1; return; }
  "$PY" -c "import torch,sys; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)['completed_iterations']))" "$ck" 2>/dev/null || echo -1
}

fit_tag() {  # fit_tag TAG DATASET Z0 Z1 STEPS TRACK_START [EXTRA_JSON [SCROLL_SPEC]]  (fit recipe via $QUIET_DIR/quiet_super.py)
  local tag=$1 ds=$2 z0=$3 z1=$4 steps=$5 ts=$6 extra=${7:-} spec=${8:-}
  local specarg=(); [ -n "$spec" ] && specarg=(--scroll-spec "$spec")
  while pgrep -f "quiet_super.py $tag " > /dev/null; do sleep 30; done   # a pre-started (OVERLAP) fit of this tag
  if [ "$(ckpt_iters "$tag")" -ge "$steps" ] 2>/dev/null; then echo "fit $tag: final checkpoint present"; return 0; fi
  local ov
  ov=$("$PY" - "$z0" "$z1" "$ts" "$extra" <<'PYEOF'
import json, sys
z0, z1, ts, ex = sys.argv[1:5]
o = {"z_begin": int(z0), "z_end": int(z1), "input_use_verified_patches": False, "input_use_gradient_magnitude": False,
     "input_use_tracks": True, "loss_start_track_dt": int(ts)}
if ex.strip():
    o.update(json.loads(ex))
print(json.dumps(o))
PYEOF
)
  echo "fit $tag $ds steps=$steps overrides=$ov"
  if [ "$DRY" = 1 ]; then $MOCK fit "$tag" "$steps" "$ts" ${spec:+"$spec"}; return $?; fi
  mkdir -p "$QW/logs"
  (cd "$QW" && "$PY" -u "$LQ/quiet_super.py" "$tag" "$ds" "$steps" none "$ov" --start-max "$GPU_START_MAX_MB" \
      --kill "$GPU_KILL_MB" --idle-util 101 --idle-s 5 --max-kills 3 ${specarg[@]+"${specarg[@]}"})
  local rc=$?
  tail -3 "$QW/logs/fit_$tag.super" 2>/dev/null
  [ "$(ckpt_iters "$tag")" -ge "$steps" ] 2>/dev/null || { echo "fit $tag ended without a final checkpoint (rc=$rc)"; return 2; }
}

sense_unknown() {  # DS : the screening config has no z-direction for it (z_top_to_bottom = -) and SENSE_MODE allows fitting
  [ "${ZD[$1]:-}" = "-" ] && [ "$SENSE_MODE" != fail ]
}

sense_list() {  # senses to fit under SENSE_MODE (lower case)
  case $SENSE_MODE in both) echo "cw acw" ;; cw) echo cw ;; acw) echo acw ;; esac
}

pick_sense() {  # pick_sense TAG DATASET_DIR [STEPS [PICK_SENSE_ARGS...]] : quiet/pick_sense.py (rule: RUNBOOK.md); rc 6 = refused, 7 = validation failed
  local tag=$1 ds=$2 steps=${3:-10000}; shift $(( $# > 3 ? 3 : $# ))
  "$PY" "$LQ/pick_sense.py" --tag "$tag" --dataset "$ds" --steps "$steps" --mode "$SENSE_MODE" \
      ${SENSE_WINDOW:+--window "$SENSE_WINDOW"} ${SENSE_MIN_REL_DIFF:+--min-rel-diff "$SENSE_MIN_REL_DIFF"} "$@"
}

flat_path() { echo "$RW/$1/flat/$1.tifxyz"; }
snap_path() { echo "$RW/$1/flat/snap/$1.tifxyz"; }

snap_flat() {  # snap_flat SCROLL TAG : move the flat mesh onto the published m7 sheet prediction (opt-in, SNAP=1)
  # Uses the PHerc0343 winner's snap_meshes.py (MIT, copied with its notice to py/snap_meshes_nieuwlaar.py): every
  # vertex moves along its normal to the nearest predicted-sheet run centre within +-15 vx; jumps are rejected and the
  # offset field is smoothed. Measured effect: FINDINGS section 8.
  local s=$1 tag=$2 src out m7
  src=$(flat_path "$tag"); out=$(dirname "$(snap_path "$tag")")
  [ -f "$out/$tag.tifxyz/x.tif" ] && { echo "snap $tag: exists"; return 0; }
  if [ "$DRY" = 1 ]; then $MOCK snap "$src" "$out/$tag.tifxyz"; return $?; fi
  m7=$("$PY" "$CLOUD/py/m7path.py" "$s") || return 1
  mkdir -p "$out/in"; ln -sfn "$src" "$out/in/$tag.tifxyz"; echo "$tag.tifxyz" > "$out/windings.txt"
  FSSPEC_S3_ANON=true SURF_ZARR="$m7" "$PY" -u "$CLOUD/py/snap_meshes_nieuwlaar.py" "$out/in" "$s" "$out" \
      --windings "$out/windings.txt" --level 0 --procs "${SNAP_PROCS:-16}"
  [ -f "$out/$tag.tifxyz/x.tif" ]
}

flatten_tag() {  # flatten_tag TAG DATASET [EXPECT_IT [VOXEL_UM]]  (official flatten via $READ_DIR/read_flatten.py)
  local tag=$1 ds=$2 exp=${3:-10000} vox=${4:-8.64} out; out=$(flat_path "$1")
  [ -f "$out/x.tif" ] && { echo "flatten $tag: exists"; return 0; }
  if [ "$DRY" = 1 ]; then $MOCK flatten "$out" "$ds" "$vox"; return $?; fi
  mkdir -p "$WORK/g1g2" "$(dirname "$out")"
  (cd "$WORK/g1g2" && "$PYX" -u "$LR/read_flatten.py" "$QRUNS/$tag/checkpoint_fitted.ckpt" \
      "$out" "$ds/umbilicus.json" "$exp" "$vox")
  [ -f "$out/x.tif" ]
}

prune_ct() {  # prune_ct SCROLL : delete the CT chunk cache of a finished scroll (public data, re-downloadable)
  [ "$PRUNE_CT" = 1 ] || { echo "PRUNE_CT=0: CT cache of $1 kept"; return 0; }
  local d
  for d in "$DATA"/s3/"$1"/volumes/*/0; do
    [ -d "$d" ] || continue
    local gb; gb=$(du -s --block-size=1G "$d" 2>/dev/null | cut -f1)
    find "$d" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -exec rm -rf {} +
    echo "pruned CT cache $d (~${gb} GB; .zarray and the absent list kept)"
  done
}
