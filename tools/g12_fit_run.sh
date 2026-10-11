#!/bin/bash
# G1 spiral fit on PHerc1447 with the official villa fit_spiral.py.
# usage: g12_fit_run.sh TAG Z_BEGIN Z_END STEPS [TRITON(0/1)]
#   e.g. g12_fit_run.sh g1_fit30k 13100 14100 30000      (band 1, as measured)
#        g12_fit_run.sh g3_fit2   11000 12350 30000      (band 2, the replication)
# Recipe (tracks OFF): patches/shell off, dense_spacing_mode grad_mag with weight 0; inputs = automatic umbilicus +
# lasagna normals + grad_mag. NOTE: the published PHerc0826 First Letters run used tracks ON (villa 6847063, where
# input_use_tracks still defaulted to true); tracks were not published for PHerc1447. See SUBMISSION.md, limitations.
# Env: VC_ROOT (default ~/vesuvius), SPIRAL_FIT_DIR (villa spiral-fitting), SPIRAL_PY (python with the spiral-fitting
# requirements), DATASET (default $VC_ROOT/data/spiral/PHerc1447).
set -e
TAG=$1; ZB=$2; ZE=$3; STEPS=$4; TRI=${5:-0}
VC_ROOT=${VC_ROOT:-$HOME/vesuvius}
VC_WORK=${VC_WORK:-$VC_ROOT/work}
SPIRAL_FIT_DIR=${SPIRAL_FIT_DIR:-${VILLA_DIR:-$VC_ROOT/repos/villa}/spiral-fitting}
SPIRAL_PY=${SPIRAL_PY:-python}
D=${DATASET:-$VC_ROOT/data/spiral/PHerc1447}
W=$VC_WORK/g1g2
mkdir -p "$W/runs" "$W/cache"
export FIT_SPIRAL_TRITON=$TRI
export WANDB_MODE=disabled
export FIT_SPIRAL_RUN_DIR=$W/runs/$TAG
export FIT_SPIRAL_CACHE_DIR=$W/cache
export FIT_SPIRAL_CONFIG_OVERRIDES="{\"z_begin\": $ZB, \"z_end\": $ZE, \"optimizer_num_training_steps\": $STEPS,
 \"input_use_verified_patches\": false, \"input_use_tracks\": false, \"input_use_fibers\": false,
 \"input_use_fiber_directions\": false, \"input_use_pcl_absolute\": false, \"input_use_pcl_relative\": false,
 \"input_use_pcl_same_winding\": false, \"input_use_pcl_drawn_control_points\": false,
 \"input_use_winding_inference\": false, \"input_use_outer_shell\": false,
 \"input_use_normals\": true, \"input_use_gradient_magnitude\": true,
 \"dense_spacing_mode\": \"grad_mag\", \"loss_weight_dense_spacing\": 0, \"loss_weight_shell_outer\": 0}"
cd "$SPIRAL_FIT_DIR"
echo "START $(date -Is) TAG=$TAG z=[$ZB,$ZE) steps=$STEPS triton=$TRI"
exec "$SPIRAL_PY" -u fit_spiral.py --dataset "$D"
