#!/bin/bash
# calibrate_343.sh -- one-command PHerc 343 calibration of the ensemble recipe (run 2026-10-11: PASS on all 18 letters; see FINDINGS §1.7).
#
# Needs: his release in REL (concat_w047-w048_R5B2_z9500-11000.{json,png,tifxyz/},
# w047_R5B2_z9500-11000.tifxyz/), the missing weights (ink9_s43 step-075000, dense_native safetensors + config),
# the 343 CT chunks under the joined mesh (render66 fetches them only with ENS_ALLOW_FETCH=1), the GPU (via
# GPU_LOCK) and ~2.5 GB free disk (66 x 2560 x 6720 render ~0.7-1.1 GB, shuffled copy, maps).
#
# usage:  calibrate_343.sh REL_DIR [cuda|cpu]   (REL_DIR = the PHerc 343 First Letters release, e.g. $VC_ROOT/ext/pherc343-first-letters)   env: FULL=1 adds the full d=-15..15 sweep afterwards
#
# Stage 1 (his surface): regions (geometry only) -> render66 of his full 2560 x 6720 canvas, his convention (no flip,
#   +d outward; outward fraction checked) -> 4 models x d=-6..6:2, reading side (reverse), tile phases 0,0 and
#   40,200 merged by his w047 rule -> ensemble (one stretch per model, z-mean) -> score + sabotage (depth-shuffled
#   render, d=-2,0,2) -> displays (yellow over dimmed CT d=+1; letters crop, nulls, line-scale strips).
# Stage 2 (OUR q343 surface): his letter/null boxes mapped through 3D xyz (stage2_q343.py); exits 10 when our
#   surface does not reach his letters (expected for q343_trk: z 8525-9467 vs his 9515-10998).
#
# PRE-REGISTERED PASS (2026-10-10; see score_regions.py): letter-box mean ensemble z (dmean over d=-6..6:2) above
# every null set's; depth-shuffled sabotage advantage < 25 % of the real one; line-scale strips show the letters
# to an observer (recorded by hand in work/calib343/observer.txt).
set -euo pipefail
REL=${1:?usage: calibrate_343.sh REL_DIR [cuda|cpu]}
DEV=${2:-cuda}
L="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # tools/ensemble
VC_ROOT=${VC_ROOT:-$HOME/vesuvius}                       # see tools/vcpaths.py
PY=${PY:-$VC_ROOT/.venv/bin/python}
W=${WORKDIR:-${VC_WORK:-$VC_ROOT/work}/ensemble/calib343}
NAME=concat_w047-w048_R5B2_z9500-11000
DEPTHS="${DEPTHS:--6..6:2}"          # env overrides are for dry runs only
SAB_DEPTHS="${SAB_DEPTHS:--2..2:2}"
MODELS="${MODELS:-ensemble4}"
NSIGN="${NSIGN:-keep}"                # his VC3D render: no flip (keep)
mkdir -p $W
exec > >(tee -a $W/calibrate.log) 2>&1
echo "== calibrate_343 start $(date -Is) REL=$REL DEV=$DEV"
# his repo layout (outputs/, inputs/) or a flat folder
if [ -d "$REL/outputs" ]; then RO=$REL/outputs; else RO=$REL; fi
for f in $NAME.json $NAME.tifxyz/x.tif w047_R5B2_z9500-11000.tifxyz/x.tif; do
  [ -e "$RO/$f" ] || { echo "missing release file $RO/$f"; exit 2; }
done
# 0. weights: fail clearly before any compute
MODELS=$MODELS ENS_DIR=$L $PY - <<'EOF'
import os, sys; sys.path.insert(0, os.environ['ENS_DIR'])
from ens import adapters as A
import os
mm = os.environ.get('MODELS', 'ensemble4'); mm = A.ENSEMBLE4 if mm == 'ensemble4' else mm.split(',')
bad = {m: s for m, s in A.status().items() if m in mm and s != 'ready'}
if bad:
    print('weights missing:'); [print(' ', m, s) for m, s in bad.items()]; sys.exit(3)
print('weights ready:', mm)
EOF
if [ "$DEV" = cuda ]; then RUN="bash $L/sweep_locked.sh calib343"; else RUN="$PY $L/depth_sweep.py"; fi

# 1. regions + merge mask (his files only, no ink read)
[ -f $W/regions.json ] || $PY $L/calib343_regions.py --rel $REL --out $W/regions.json

# 2. render66 of his full canvas (his VC3D call has no flip; +d must point outward)
if [ ! -d $W/$NAME.zarr ]; then
  $PY $L/render66.py --scroll PHerc0343 --tifxyz $RO/$NAME.tifxyz --normal-sign $NSIGN --out $W/$NAME.zarr --procs 8
fi
$PY - <<EOF
import sys; sys.path.insert(0, '$L')
from ens import common as C
_, at = C.open_render('$W/$NAME.zarr'); r = at['ensemble_render']
print('outward fraction of +d:', r['outward_fraction_plusd'], '(his sidecar: normal_outward_fraction 0.9973)')
if r['outward_fraction_plusd'] < 0.8: sys.exit('normal sign disagrees with his convention; stop and check')
EOF
[ -d $W/${NAME}_shuffled.zarr ] || $PY $L/tools/shuffle_layers.py $W/$NAME.zarr $W/${NAME}_shuffled.zarr --seed 0

# 3. depth sweeps (reading side = reverse), tile-phase merge by his w047 rule
$RUN --render $W/$NAME.zarr --models $MODELS --depths=$DEPTHS --order reverse --phase 0,0 --phase 40,200 \
     --merge mask:$W/merge_w047.npy --out $W/sweep --device $DEV
$RUN --render $W/${NAME}_shuffled.zarr --models $MODELS --depths=$SAB_DEPTHS --order reverse --phase 0,0 \
     --out $W/sweep_shuffled --device $DEV

# 4. ensembles (one stretch per model over the depths, z-mean of the 4 models; depth mean and max)
$PY $L/ensemble_maps.py --render $W/$NAME.zarr --maps $W/sweep --models $MODELS --tag merged --stat-depths=$DEPTHS \
     --write-depths=$DEPTHS --combine mean --out $W/ens --name $NAME
$PY $L/ensemble_maps.py --render $W/$NAME.zarr --maps $W/sweep --models $MODELS --tag merged --stat-depths=$DEPTHS \
     --write-depths=$DEPTHS --combine max --out $W/ens_max --name $NAME --no-png
$PY $L/ensemble_maps.py --render $W/${NAME}_shuffled.zarr --maps $W/sweep_shuffled --models $MODELS --stat-depths=$SAB_DEPTHS \
     --write-depths=$SAB_DEPTHS --combine mean --out $W/ens_shuffled --name $NAME --no-png

# 5. pre-registered score
$PY $L/score_regions.py --ens $W/ens --name $NAME --regions $W/regions.json --primary dmean \
     --sabotage-ens $W/ens_shuffled --out $W/score.json
$PY $L/score_regions.py --ens $W/ens_max --name $NAME --regions $W/regions.json --primary dmax --out $W/score_dmax.json

# 6. displays in reading orientation (his flips), CT at d=+1
CROPS=$($PY - <<EOF
import json; r = json.load(open('$W/regions.json')); b = r['letters_raw']
r0 = max(0, min(x[0] for x in b) - 150); r1 = max(x[1] for x in b) + 150
c0 = max(0, min(x[2] for x in b) - 150); c1 = max(x[3] for x in b) + 150
print(f'{r0}:{r1},{c0}:{c1}')
EOF
)
FL=$($PY -c "import json; r=json.load(open('$W/regions.json')); print(('--flip-ud ' if r['flip_ud'] else '')+('--flip-lr' if r['flip_lr'] else ''))")
$PY $L/display.py --render $W/$NAME.zarr --ens-dir $W/ens --name $NAME --ink dmean --ct-d 1 $FL \
     --out $W/display_full_dmean.png
$PY $L/display.py --render $W/$NAME.zarr --ens-dir $W/ens --name $NAME --ink dmean --ct-d 1 $FL --crop $CROPS \
     --out $W/display_letters_dmean.png --strips $W/strips_letters --strip-h 450
for d in 0 2; do
  $PY $L/display.py --render $W/$NAME.zarr --ens-dir $W/ens --name $NAME --ink-d $d --ct-d 1 $FL --crop $CROPS \
       --out $W/display_letters_d$d.png
done
echo "OBSERVER CHECK: open $W/strips_letters/*.png and record 'letters visible: yes/no' in $W/observer.txt"

# 7. stage 2 on OUR q343 surface (exit 10 = our surface does not reach his letters)
set +e
$PY $L/stage2_q343.py --rel $REL --regions $W/regions.json --out $W/stage2 --run --device $DEV --depths=$DEPTHS --sab-depths=$SAB_DEPTHS --models $MODELS --margin ${S2_MARGIN:-256}
rc=$?
set -e
echo "stage2 rc=$rc"

# 8. optional full 31-depth sweep (his exact depth set) for the d=+1 image
if [ "${FULL:-0}" = 1 ]; then
  $RUN --render $W/$NAME.zarr --models $MODELS --depths=-15..15 --order reverse --phase 0,0 --phase 40,200 \
       --merge mask:$W/merge_w047.npy --out $W/sweep --device $DEV
  $PY $L/ensemble_maps.py --render $W/$NAME.zarr --maps $W/sweep --models $MODELS --tag merged --stat-depths=-15..15 \
       --write-depths=-15..15 --combine mean --out $W/ens_full --name $NAME
  $PY $L/display.py --render $W/$NAME.zarr --ens-dir $W/ens_full --name $NAME --ink-d 1 --ct-d 1 $FL \
       --out $W/display_full_d+1.png
fi
echo "== calibrate_343 done $(date -Is)"
