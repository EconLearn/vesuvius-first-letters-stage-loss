#!/bin/bash
# bootstrap.sh - idempotent pod setup for the Vesuvius First Letters screening run (RunPod RTX 4090, Ubuntu 24.04 image).
# Run AFTER this repository (or the tarball from pack_code.sh) is on the pod, with VC_ROOT pointing at the pod's
# persistent volume (see RUNBOOK.md):
#     VC_ROOT=/path/on/the/volume bash tools/cloud/bootstrap.sh [--dry-run] [--check-locks] [--with-hecate]
# What it does (each step skipped when already done; log: $VC_ROOT/bootstrap.log):
#   1 layout   $VC_ROOT directories, .POD marker (the disk-guard shim in pod/sitecustomize.py looks for it)
#   2 apt      build-essential git rsync curl tmux (g++ >= 13 needed for the vc_spiral C++23 extension)
#   3 driver   nvidia-smi; the pinned torch 2.12.1 is the CUDA 13.0 build -> driver >= 580 (else cu128 fallback, flagged)
#   4 uv       uv 0.12.21 (pip, PyPI) + CPython 3.14.7, both under $POD_TOOLS (default $VC_ROOT/pod_tools: they live on the volume)
#   5 venvs    .venv = lock/req-main.txt (exact freeze) + editable villa vesuvius (--no-deps);
#              .venv-x2 = lock/req-x2.txt layered over .venv via zz_base_venv.pth; disk-guard shim
#   6 villa    ScrollPrize/villa @ e0bbb8b4 (blob-less, sparse: vesuvius + spiral-fitting + lasagna) -> $VILLA_DIR
#   7 native   vc_spiral extensions built in $SPIRAL_FIT_DIR (default $VILLA_DIR/spiral-fitting; cmake RelWithDebInfo) + import check
#   8 release  Nieuwlaar/pherc343-first-letters @ e8370f1f -> $PHERC343_REPO (default $VC_EXT/pherc343-first-letters),
#              sha256 of outputs/ checked; 8b: letter/null boxes (calib343_regions.py) derived from its geometry
#   9 weights  HF downloads from lock/models.tsv with sha256 checks (Hecate only with --with-hecate)
#  10 ct       CT cache dir on the pod volume (NVMe) + .zarray of PHerc0343 and the scrolls of screen_scrolls.tsv (public S3)
#  11 checks   torch.cuda, weights status, fit_spiral --help, cost guard init
# The fit supervisor / input builders ($QUIET_DIR, default tools/cloud/quiet), the flatten wrapper ($READ_DIR, default
# tools/cloud/read) and the GPU lock wrapper ($GPU_LOCKED, default tools/cloud/gpu/gpu_locked.sh) are part of this
# repository; if QUIET_DIR / READ_DIR / GPU_LOCKED point elsewhere and a file is missing, a WARNING is printed.
set -uo pipefail
DRY=0; CHECK_LOCKS=0; HECATE=0
for a in "$@"; do case $a in --dry-run) DRY=1;; --check-locks) CHECK_LOCKS=1;; --with-hecate) HECATE=1;; esac; done
HERE=$(cd "$(dirname "$0")" && pwd)                    # tools/cloud
CODE=$(cd "$HERE/../.." && pwd)                        # repository root (checkout or unpacked code tarball)
TOOLS=$CODE/tools
ROOT=${VC_ROOT:-$HOME/vesuvius}                        # see tools/vcpaths.py
DATA=${VC_DATA:-$ROOT/data}; WORK=${VC_WORK:-$ROOT/work}; CKPT=${VC_CKPT:-$ROOT/ckpt}; EXT=${VC_EXT:-$ROOT/ext}
VILLA=${VILLA_DIR:-$ROOT/repos/villa}
SF=${SPIRAL_FIT_DIR:-$VILLA/spiral-fitting}
LQ=${QUIET_DIR:-$HERE/quiet}; LR=${READ_DIR:-$HERE/read}; GPU_LOCKED=${GPU_LOCKED:-$HERE/gpu/gpu_locked.sh}
QW=${QUIET_WORK:-$WORK/quiet}; QRUNS=${QUIET_RUNS:-$QW/runs}; RW=${READ_WORK:-$WORK/read}
RESULTS=${RESULTS_DIR:-${VC_OUT:-$ROOT/out}/cloud}
SCREEN_CONFIG=${SCREEN_CONFIG:-$HERE/screen_scrolls.tsv}
export VC_ROOT=$ROOT VC_DATA=$DATA VC_WORK=$WORK VC_CKPT=$CKPT VC_EXT=$EXT VILLA_DIR=$VILLA SPIRAL_FIT_DIR=$SF SCREEN_CONFIG
PODTOOLS=${POD_TOOLS:-$ROOT/pod_tools}
export UV_PYTHON_INSTALL_DIR=$PODTOOLS/uv-python UV_CACHE_DIR=$ROOT/.uv-cache UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
UV=$PODTOOLS/bin/uv
PYV=3.14.7; UVV=0.12.21
VILLA_COMMIT=e0bbb8b40a2db58b1d71864f286eb85717e59e64
REL_COMMIT=e8370f1f88a211166056df62ea6fcfffe4a1063e
STATE=${POD_STATE:-$ROOT/cloud_state}
LOCK=$HERE/lock
if [ "$DRY" = 1 ]; then
  STATE=${POD_STATE:-$(mktemp -d)/cloud_state}; mkdir -p "$STATE/logs"; LOG=$STATE/logs/bootstrap.log
else
  mkdir -p "$ROOT" 2>/dev/null; LOG=$ROOT/bootstrap.log
fi
say() { echo "$(date -Is) [bootstrap] $*" | tee -a "$LOG"; }
run() { if [ "$DRY" = 1 ]; then echo "  [dry] $*" | tee -a "$LOG"; else say "+ $*"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; fi; }
die() { say "FATAL: $*"; exit 1; }

say "bootstrap start (dry=$DRY) ROOT=$ROOT CODE=$CODE"
# ---- static checks (also in --dry-run): the kit is complete and the locks parse
for f in req-main.txt req-x2.txt models.tsv PINS.txt; do [ -s "$LOCK/$f" ] || die "missing $LOCK/$f"; done
NMAIN=$(grep -cvE '^\s*(#|$)' "$LOCK/req-main.txt"); NX2=$(grep -cvE '^\s*(#|$)' "$LOCK/req-x2.txt")
grep -qE '^torch==2\.12\.1$' "$LOCK/req-main.txt" || die "req-main.txt does not pin torch==2.12.1"
grep -vE '^\s*(#|$)' "$LOCK/req-main.txt" "$LOCK/req-x2.txt" | grep -vE ':[A-Za-z0-9_.-]+==[^ ]+$' && die "unpinned lines in the locks"
say "locks: $NMAIN pinned packages (main), $NX2 (x2)"
for f in tools/ensemble/depth_sweep.py tools/ensemble/render66.py tools/ensemble/stage2_q343.py tools/ensemble/score_matched.py \
         tools/ensemble/calib343_regions.py tools/vcpaths.py tools/g12lib.py tools/vlib/guards.py tools/cloud/screen_scrolls.tsv; do
  [ -e "$CODE/$f" ] || die "code incomplete: $f"
done
say "code: required repository files present"
for f in "$LQ/quiet_super.py" "$LQ/build_normals_band.py" "$LQ/tracks_subset.py" "$LQ/m7normals.py" \
         "$LQ/fetch_lasagna_band.py" "$LQ/umb_auto.py" "$LQ/umb_smooth.py" "$LQ/trk_loadcheck.py" \
         "$LR/read_flatten.py" "$GPU_LOCKED" "$(dirname "$GPU_LOCKED")/pending_resume.py"; do
  [ -e "$f" ] || say "WARNING: missing: $f (jobs 2-3 and inputs.sh need it; QUIET_DIR / READ_DIR / GPU_LOCKED point elsewhere? see RUNBOOK.md)"
done
if [ "$CHECK_LOCKS" = 1 ]; then   # every pinned package exists for CPython 3.14 / manylinux x86_64 (PyPI metadata only)
  UVC=$(command -v uv || echo "$UV")
  for f in req-main req-x2; do
    "$UVC" pip compile --quiet --python-version 3.14 --python-platform x86_64-manylinux_2_39 --no-deps         -o /dev/null "$LOCK/$f.txt" >> "$LOG" 2>&1 && say "lock check: $f resolves for cp314/manylinux" || die "lock check failed: $f (see $LOG)"
  done
fi

# ---- 1 layout
# (spiral-fitting is used in place inside the villa checkout, where its ../vesuvius package directory already exists)
run mkdir -p "$ROOT"
run touch "$ROOT/.POD"
run mkdir -p "$STATE/logs" "$DATA/s3" "$QRUNS" "$QW/logs" "$QW/out" "$QW/tmp" "$RW" "$WORK/g1g2" "$RESULTS" "$PODTOOLS/bin"

# ---- 2 apt
NEED=""
for c in g++ git rsync curl tmux flock pgrep; do command -v $c > /dev/null || NEED="$NEED $c"; done
if [ -n "$NEED" ]; then
  say "apt needed for:$NEED"
  run apt-get update -qq
  run env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential git rsync curl tmux util-linux procps ca-certificates
fi
. /etc/os-release 2>/dev/null
GCCV=$(g++ -dumpversion 2>/dev/null | cut -d. -f1); GCCV=${GCCV:-0}
say "OS ${PRETTY_NAME:-unknown}, g++ major $GCCV"
[ "$DRY" = 1 ] || [ "$GCCV" -ge 13 ] || die "g++ >= 13 required for vc_spiral (C++23). Use an Ubuntu 24.04 image (RUNBOOK step 1)."

# ---- 3 driver
TORCH_FALLBACK=""
if command -v nvidia-smi > /dev/null; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
  say "GPU: $GPU, driver $DRV"
  [ "${DRV%%.*}" -ge 580 ] 2>/dev/null || { TORCH_FALLBACK=cu128; say "WARNING driver $DRV < 580: the pinned CUDA-13 torch cannot run; installing torch 2.12.1 from the cu128 index instead (NOT the pinned build; flagged in BOOTSTRAP_OK)"; }
else
  say "nvidia-smi not found (dry run or CPU host)"
fi

# ---- 4 uv + python
if [ ! -x "$UV" ]; then
  if command -v python3 > /dev/null && run python3 -m pip install -q --target "$PODTOOLS/uvpkg" "uv==$UVV"; then
    run ln -sf "$PODTOOLS/uvpkg/bin/uv" "$UV"
  else
    run sh -c "curl -LsSf https://astral.sh/uv/$UVV/install.sh | env UV_INSTALL_DIR=$PODTOOLS/bin UV_NO_MODIFY_PATH=1 sh"
  fi
fi
[ "$DRY" = 1 ] || "$UV" --version | grep -q "$UVV" || say "WARNING: uv version $("$UV" --version) != $UVV"
run "$UV" python install "$PYV"

# ---- 5 venvs
if [ ! -x "$ROOT/.venv/bin/python" ]; then run "$UV" venv "$ROOT/.venv" --python "$PYV"; fi
if [ ! -f "$ROOT/.venv/.lock_installed" ]; then
  run "$UV" pip install --python "$ROOT/.venv/bin/python" -r "$LOCK/req-main.txt" || die "main venv install failed"
  if [ -n "$TORCH_FALLBACK" ]; then
    run "$UV" pip install --python "$ROOT/.venv/bin/python" --reinstall-package torch --reinstall-package torchvision \
        --index-url "https://download.pytorch.org/whl/$TORCH_FALLBACK" --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match "torch==2.12.1" "torchvision==0.27.1" || die "torch fallback install failed"
  fi
  [ "$DRY" = 1 ] || touch "$ROOT/.venv/.lock_installed"
fi
if [ ! -x "$ROOT/.venv-x2/bin/python" ]; then run "$UV" venv "$ROOT/.venv-x2" --python "$PYV"; fi
if [ ! -f "$ROOT/.venv-x2/.lock_installed" ]; then
  run "$UV" pip install --python "$ROOT/.venv-x2/bin/python" -r "$LOCK/req-x2.txt" --no-deps || die "x2 venv install failed"
  # --no-deps: with dependencies, torchdiffeq pulled torch 2.14.1/triton 3.8 into .venv-x2,
  # shadowing the pinned torch 2.12.1 in .venv (req-x2 is a full freeze, so nothing is missing)
  [ "$DRY" = 1 ] || touch "$ROOT/.venv-x2/.lock_installed"
fi
SP=$ROOT/.venv/lib/python3.14/site-packages; SPX=$ROOT/.venv-x2/lib/python3.14/site-packages
run sh -c "echo '$SP' > '$SPX/zz_base_venv.pth'"
run cp -f "$HERE/pod/sitecustomize.py" "$SP/sitecustomize.py"

# ---- 6 villa
if [ ! -d "$VILLA/.git" ]; then
  run git clone --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git "$VILLA"
  run git -C "$VILLA" sparse-checkout set --cone vesuvius spiral-fitting lasagna
fi
if [ "$DRY" = 1 ] || [ "$(git -C "$VILLA" rev-parse HEAD 2>/dev/null)" != "$VILLA_COMMIT" ]; then
  run git -C "$VILLA" checkout -q "$VILLA_COMMIT" || die "villa checkout failed"
fi
if [ ! -f "$ROOT/.venv/.villa_installed" ]; then
  run "$UV" pip install --python "$ROOT/.venv/bin/python" --no-deps -e "$VILLA/vesuvius" || die "villa vesuvius install failed"
  [ "$DRY" = 1 ] || touch "$ROOT/.venv/.villa_installed"
fi

# ---- 7 native extensions (built in the spiral-fitting directory, sources only)
if [ "$DRY" = 0 ]; then
  for f in "$SF/fit_spiral.py" "$SF/build_track_crossings.py" "$SF/cpp/CMakeLists.txt"; do
    [ -e "$f" ] || die "spiral-fitting incomplete: $f (SPIRAL_FIT_DIR=$SF)"
  done
fi
if [ "$DRY" = 1 ] || ! (cd "$SF" && "$ROOT/.venv-x2/bin/python" -c "import vc_spiral.track_store, vc_spiral.track_crossings, vc_spiral.spiral_sampling, vc_spiral.surface_index" 2>/dev/null); then
  run "$ROOT/.venv-x2/bin/cmake" -S "$SF/cpp" -B "$SF/build" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython_EXECUTABLE="$ROOT/.venv-x2/bin/python" || die "cmake configure failed"
  run "$ROOT/.venv-x2/bin/cmake" --build "$SF/build" -j "$(nproc)" || die "vc_spiral build failed"
  run sh -c "cp -f '$SF'/build/vc_spiral/*.so '$SF/vc_spiral/'"
  [ "$DRY" = 1 ] || (cd "$SF" && "$ROOT/.venv-x2/bin/python" -c "import vc_spiral.track_store, vc_spiral.track_crossings, vc_spiral.spiral_sampling, vc_spiral.surface_index; print('vc_spiral OK')") | tee -a "$LOG" || die "vc_spiral import failed"
fi

# ---- 8 his release (data only; none of its scripts are run)
R=${PHERC343_REPO:-$EXT/pherc343-first-letters}
if [ ! -d "$R/.git" ]; then run git clone -q https://github.com/Nieuwlaar/pherc343-first-letters "$R"; fi
run git -C "$R" checkout -q "$REL_COMMIT"
if [ "$DRY" = 0 ]; then (cd "$R/outputs" && sha256sum -c --quiet SHA256SUMS) || die "release sha256 mismatch"; say "release sha256 OK"; fi
# 8b letter boxes + pre-registered null boxes on his surface (geometry only; no ink map is read)
REG=${CALIB343_REGIONS:-$WORK/ensemble/calib343/regions.json}
if [ "$DRY" = 1 ] || [ ! -s "$REG" ]; then
  run mkdir -p "$(dirname "$REG")"
  run "$ROOT/.venv/bin/python" "$TOOLS/ensemble/calib343_regions.py" --rel "$R" --out "$REG" || die "calib343_regions failed"
fi

# ---- 9 weights (sha256 from the release's ink/models.tsv, also in lock/models.tsv)
fetch() {  # KEY DEST
  local url sha
  url=$(awk -F'\t' -v k="$1" '$1==k {print $6}' "$LOCK/models.tsv"); sha=$(awk -F'\t' -v k="$1" '$1==k {print $4}' "$LOCK/models.tsv")
  [ -n "$url" ] && [ -n "$sha" ] || die "models.tsv has no url/sha for $1"
  if [ -f "$2" ] && [ "$DRY" = 0 ] && echo "$sha  $2" | sha256sum -c --quiet - 2>/dev/null; then say "weights ok: $2"; return 0; fi
  run mkdir -p "$(dirname "$2")"
  run curl -L --fail --retry 5 --retry-delay 5 -C - -o "$2.part" "$url" || die "download failed: $url"
  if [ "$DRY" = 0 ]; then echo "$sha  $2.part" | sha256sum -c --quiet - || die "sha256 mismatch for $1"; mv "$2.part" "$2"; say "weights ok: $2"; fi
}
fetch dnative_public_weights "$EXT/weights/dense_native/dense_native_016000.safetensors"
fetch dnative_public_config "$EXT/weights/dense_native/train_dense_native.json"
fetch hybrid_3d2d-seed42 "$CKPT/ink_9um/hybrid_3d2d-seed42/step-075000.pth"
fetch hybrid_3d2d-seed43 "$EXT/weights/ink_9um_seed43/step-075000.pth"
if [ "$HECATE" = 1 ]; then
  fetch hecate "$EXT/weights/hecate/hecate_9.6um.pth"; fetch hecate_code "$EXT/weights/hecate/hecate.py"
fi

# ---- 10 CT cache on the pod volume + volume metadata (PHerc0343 and every scroll with volume metadata in the config)
VOLS=$(awk -F'\t' '$1 !~ /^#/ && $1 != "dataset" && $8 != "-" {print $2}' "$SCREEN_CONFIG" | sort -u | tr '\n' ' ')
run "$ROOT/.venv/bin/python" "$HERE/py/volumes.py" PHerc0343 $VOLS

# ---- 11 checks
if [ "$DRY" = 0 ]; then
  "$ROOT/.venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | tee -a "$LOG"
  "$ROOT/.venv/bin/python" "$TOOLS/ensemble/depth_sweep.py" --status | tee -a "$LOG"
  (cd "$SF" && "$ROOT/.venv-x2/bin/python" fit_spiral.py --help 2>&1 | head -3) | tee -a "$LOG"
  "$ROOT/.venv/bin/python" -c "import os, shutil; p=os.environ.get('POD_GUARD_PATH') or os.environ['VC_ROOT']; u=shutil.disk_usage(p); print('disk guard path', p, 'free =', round(u.free/1e9), 'GB (pod volume; the POD_GUARD_PATH shim adds its offset)')" | tee -a "$LOG"
  POD_STATE=$STATE "$ROOT/.venv/bin/python" "$HERE/py/cost_guard.py" init | tee -a "$LOG"
  { echo "bootstrap_ok $(date -Is)"; echo "torch_fallback=${TORCH_FALLBACK:-none}"; "$UV" --version; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$STATE/BOOTSTRAP_OK"
fi
say "bootstrap done (dry=$DRY). Next: bash $HERE/jobs/job1_timing.sh (see RUNBOOK)"
