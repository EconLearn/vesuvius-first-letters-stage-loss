#!/bin/bash
# JOB 1 - timing check on the 4090 (about 20 min of measurement; the 500-step fit waits for the PHerc0343x inputs,
# which stream in the background meanwhile and are needed by job 2 anyway).
#  1. background: inputs for PHerc0343x (published lasagna band, own umbilicus, published tracks + crossings)
#  2. render a 1500 x 1540 px canvas of HIS released 343 surface around letters (CPU + public-S3 CT), k 19..47
#  3. GPU sweeps at d = -2, 0, +2 (reading side): ink9_s42 batch 16, dense_native batch 16 and 48; 1 s GPU samples
#  4. fit 500 steps on PHerc0343x z 9750-10750 (1000 slices, like the 4060 reference), track dt from step 250,
#     so both the pre- and post-track rates are measured
#  5. timing_report.py -> $STATE/timing.json: r_eff it/s vs the RTX 4060 reference (1.83-2.07), villa s/Mpx vs the
#     4060 anchors, render s/Mpx-layer, S3 MB/s, GPU utilisation, re-projected job hours/$ (guesses).
#     STOP RULE: proceed only if r_eff >= STOP_IT_S (3.0 it/s). FAIL -> exit 5 and jobs 2-3 refuse to start.
#  6. if PASS: background builds of the screening datasets that are not being uploaded (INPUT_MODE in pod.env)
JOB=job1
. "$(dirname "$0")/lib.sh"
own_group "$0" "$@"
job_lock
"$PY" "$CLOUD/py/cost_guard.py" init --budget "$BUDGET_USD" --rate "$RATE_USD_H"
start_watchdog
W=$WORK/cloud/timing; mkdir -p "$W"
NAME=concat_w047-w048_R5B2_z9500-11000
ROWS=${TIMING_ROWS:-600:2100}; COLS=${TIMING_COLS:-3900:5440}
: "${INPUT_MODE:=rebuild}"
status "JOB 1 timing check start (DRY_RUN=$DRY, INPUT_MODE=$INPUT_MODE)"

step inputs_bg_343x 0.01 bash "$CLOUD/jobs/inputs.sh" bg PHerc0343x

render_timing() {
  [ -d "$W/timing.zarr" ] && return 0
  local m b0 b1 t0 t1
  m=$("$PY" -c "import sys; sys.path.insert(0, '$CLOUD/py'); import volumes; print(volumes.mirror_dir('PHerc0343'))")
  b0=$(du -sb "$m" 2>/dev/null | cut -f1); b0=${b0:-0}; t0=$(date +%s.%N)
  if [ "$DRY" = 1 ]; then
    $MOCK render --out "$W/timing.zarr" --rows "$ROWS" --cols "$COLS" --k-range 19:48 || return 2
  else
    ENS_ALLOW_FETCH=1 OMP_NUM_THREADS=1 "$PY" "$CLOUD/py/r66.py" --scroll PHerc0343 --tifxyz "$REL/outputs/$NAME.tifxyz" \
      --rows "$ROWS" --cols "$COLS" --normal-sign keep --k-range 19:48 --out "$W/timing.zarr" --procs "$RENDER_PROCS" || return 2
  fi
  t1=$(date +%s.%N); b1=$(du -sb "$m" 2>/dev/null | cut -f1); b1=${b1:-0}
  "$PY" -c "import json,sys; json.dump(dict(secs=float(sys.argv[1])-float(sys.argv[2]), layers=29, procs=int(sys.argv[3]), fetched_bytes=int(sys.argv[4])-int(sys.argv[5]), rows='$ROWS', cols='$COLS'), open('$W/render.json','w'), indent=1)" \
     "$t1" "$t0" "$RENDER_PROCS" "$b1" "$b0"
  cat "$W/render.json"
}
step render 0.15 render_timing

sweep_timing() {  # LABEL MODEL BATCH
  [ -f "$W/sw_$1/.done" ] && return 0
  if [ "$DRY" = 1 ]; then $MOCK sweep --render "$W/timing.zarr" --models "$2" --depths=-2..2:2 --out "$W/sw_$1" || return 2
  else "$PY" "$CLOUD/py/sweep.py" --render "$W/timing.zarr" --models "$2" --depths=-2..2:2 --order reverse --phase 0,0 \
         --out "$W/sw_$1" --device "$DEVICE" --batch "$3" || return 2; fi
  touch "$W/sw_$1/.done"; cat "$W/sw_$1/timings.tsv"
}
SP=$(gpu_sampler "$W/gpu_sweep.csv")
step sweep_ink9_b16 0.1 sweep_timing ink9b16 ink9_s42 16 || { kill "$SP"; exit 2; }
step sweep_dn_b16 0.1 sweep_timing dnb16 dnative 16 || { kill "$SP"; exit 2; }
step sweep_dn_b48 0.1 sweep_timing dnb48 dnative 48 || { kill "$SP"; exit 2; }
kill "$SP" 2>/dev/null

step inputs_343x 1.5 bash "$CLOUD/jobs/inputs.sh" ensure PHerc0343x || exit 2
SP=$(gpu_sampler "$W/gpu_fit.csv")
step fit500 0.3 fit_tag t343x_500 "$DATA/spiral/PHerc0343x" 9750 10750 500 250 || { kill "$SP"; exit 2; }
kill "$SP" 2>/dev/null

step report 0.02 "$PY" "$CLOUD/py/timing_report.py" --fit-log "$QW/logs/fit_t343x_500.log" --track-start 250 \
   --render "$W/timing.zarr" --render-json "$W/render.json" --sweep "ink9_s42@b16=$W/sw_ink9b16" \
   --sweep "dnative@b16=$W/sw_dnb16" --sweep "dnative@b48=$W/sw_dnb48" --gpu "sweep=$W/gpu_sweep.csv" \
   --gpu "fit=$W/gpu_fit.csv" --rate "$RATE_USD_H" --stop-it-s "$STOP_IT_S" --out "$STATE/timing.json" || exit 2
grep -E '^(FIT|VILLA|RENDER|GPU|PROJECTION|RECOMMEND)' "$STATE/logs/job1.report.log" | while read -r l; do status "$l"; done
PASS=$("$PY" -c "import json; print(json.load(open('$STATE/timing.json'))['stop_rule']['passed'])")
if [ "$PASS" != True ]; then
  echo FAIL > "$STATE/STOP_RULE"
  status "STOP RULE FAILED: the fit runs below $STOP_IT_S it/s (see timing.json). Jobs 2-3 will refuse to start (FORCE=1 overrides). Stop the pod unless you decide otherwise."
  exit 5
fi
echo PASS > "$STATE/STOP_RULE"
status "STOP RULE PASSED"

# datasets and the upload set come from screen_scrolls.tsv (SCREEN_DEFAULT / UPLOAD_DEFAULT in lib.sh)
if [ "$INPUT_MODE" = upload ]; then
  REST=""; UPSET=" ${UPLOAD_SETS:-$UPLOAD_DEFAULT} "
  for s in $SCREEN_DEFAULT; do case "$UPSET" in *" $s "*) ;; *) REST="${REST:+$REST }$s" ;; esac; done
else REST="$SCREEN_DEFAULT"; fi
step inputs_bg_rest 0.01 bash "$CLOUD/jobs/inputs.sh" bg $REST
status "JOB 1 done. Next: bash $CLOUD/jobs/job2_stage2_343.sh"
