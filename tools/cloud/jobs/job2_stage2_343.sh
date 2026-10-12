#!/bin/bash
# JOB 2 - stage 2 on PHerc0343: OUR tracks fit of z 9500-11000, flattened, his 18 letter boxes mapped onto OUR
# surface through 3D xyz (stage2_q343.py, unchanged), ink chain with depth-shuffled sabotage, then the calibration's
# matched-null rule on our surface (stage2_matched.py -> score_matched.py, unchanged).
#
# PRE-REGISTERED (written 2026-10-11, before any stage-2 ink exists):
#   coverage gate: stage2_q343's own rule (>= 50 % of letter boxes covered); otherwise exit 10 = NOT TESTABLE.
#   PASS iff INK_EVIDENCED (score_matched, unchanged) for BOTH dense_native and the 3-model z-mean at d = -2..+2, each
#   with >= 9 scored letters in both matched nulls (interline, ring); < 9 letters -> INCONCLUSIVE.
#   Secondary, reported: the stage-1 rule (score_regions.py inside stage2_q343 --run) and the other members.
#   Observer check: someone reads the strips (record in $W/observer.txt by hand).
JOB=job2
. "$(dirname "$0")/lib.sh"
own_group "$0" "$@"
if [ "$(cat "$STATE/STOP_RULE" 2>/dev/null)" != PASS ] && [ "${FORCE:-0}" != 1 ]; then
  status "refusing: job 1 stop rule has not passed (STOP_RULE=$(cat "$STATE/STOP_RULE" 2>/dev/null || echo missing)); FORCE=1 overrides"; exit 5
fi
job_lock
"$PY" "$CLOUD/py/cost_guard.py" init --budget "$BUDGET_USD" --rate "$RATE_USD_H"
start_watchdog
DS=$DATA/spiral/PHerc0343x
TAG=q343x_trk
W=$WORK/ensemble/stage2x
FLAT=$(flat_path $TAG)
MODELS=${S2_MODELS:-ink9_s42,ink9_s43,dnative}
status "JOB 2 stage 2 on PHerc0343 start (models $MODELS; DRY_RUN=$DRY)"
est() { "$PY" -c "import json,sys; p=json.load(open('$STATE/timing.json'))['projections_GUESS']['job2_stage2_343']; print(round(p[sys.argv[1]],2))" "$1" 2>/dev/null || echo "$2"; }

step inputs 1.0 bash "$CLOUD/jobs/inputs.sh" ensure PHerc0343x || exit 2
step fit "$(est fit_1500_slices_h 1.5)" fit_tag $TAG "$DS" 9500 11000 10000 "$TRACK_START" || exit 2
step flatten 0.15 flatten_tag $TAG "$DS" 10000 "${VX[PHerc0343x]:-8.64}" || exit 2

umb_union() {  # outward-normal reference for 343 renders: existing line (z 8100-9900) + our band line (z >= 9500)
  local u=$DATA/spiral/PHerc0343/umbilicus.json
  [ -f "$u.original" ] || cp -p "$u" "$u.original"
  "$PY" - "$u.original" "$DS/umbilicus.json" "$u" <<'PYEOF'
import json, sys
a, b, out = sys.argv[1:4]
A = json.load(open(a)); B = json.load(open(b))
zmin_b = min(p['z'] for p in B['control_points'])
cps = [p for p in A['control_points'] if p['z'] < zmin_b] + B['control_points']
json.dump(dict(control_points=sorted(cps, key=lambda p: p['z']),
               provenance=dict(note='POD union for the render66 outward-normal check: original umbilicus below z %d, '
                                    'PHerc0343x band umbilicus above (job2_stage2_343.sh); original kept as .original' % zmin_b,
                               a=A.get('provenance', {}).get('method'), b=B.get('provenance', {}).get('method'))),
          open(out, 'w'), indent=1)
print('umbilicus union', len(cps), 'control points, z', cps[0]['z'], '-', cps[-1]['z'])
PYEOF
}
step umb_union 0.01 umb_union || exit 2

stage2_run() {
  if [ "$DRY" = 1 ]; then $MOCK stage2 --out "$W" ${MOCK_STAGE2_RC:+--rc $MOCK_STAGE2_RC}; return $?; fi
  ENS_ALLOW_FETCH=1 "$PY" "$LE/stage2_q343.py" --rel "$REL" --regions "$REGIONS" --ours "$FLAT" \
     --out "$W" --run --device "$DEVICE" --models "$MODELS" --depths=-6..6:2 --sab-depths=-2..2:2 --margin 256
}
S2EST=$("$PY" -c "import json; p=json.load(open('$STATE/timing.json'))['projections_GUESS']['job2_stage2_343']; print(round(p['render_h']+p['sweeps_h']+p['post_h'],2))" 2>/dev/null || echo 2.5)
if ! is_done stage2; then
  guard stage2 "$S2EST" || { status "STOP before stage2 (cost guard)"; exit 3; }
  status "START stage2 (est $S2EST h)"
  stage2_run >> "$STATE/logs/job2.stage2.log" 2>&1; rc=$?
  if [ $rc -eq 10 ] || [ $rc -eq 11 ]; then
    grep -E 'our_z_range|letters_covered|NO COVERAGE|implausibly' "$STATE/logs/job2.stage2.log" | tail -3 | while read -r l; do status "$l"; done
    echo "{\"verdict\": \"NOT TESTABLE\", \"stage2_rc\": $rc}" > "$W/PASS_NOT_TESTABLE.json"
    status "STAGE 2 NOT TESTABLE (rc=$rc: our surface does not cover >= 50 % of his letters, or the mapping scattered). Job 3 can proceed."
    exit 10
  elif [ $rc -ne 0 ]; then status "FAIL stage2 rc=$rc (log $STATE/logs/job2.stage2.log)"; exit 2; fi
  date -Is > "$STATE/done/job2__stage2"; status "DONE stage2"
fi
grep -E 'letters_covered' "$STATE/logs/job2.stage2.log" | tail -1 | while read -r l; do status "coverage: $l"; done

step matched 0.05 "$PY" "$CLOUD/py/stage2_matched.py" --stage2 "$W" --out "$W/matched" --models "$MODELS" || exit 2
grep -E '^STAGE2 VERDICT' "$STATE/logs/job2.matched.log" | tail -1 | while read -r l; do status "$l"; done

letter_crops() {  # one crop per mapped letter box (3-model z-mean, mean over d=-6..+6, CT d=+1) for the observer
  "$PY" - "$W" "$LE" <<'PYEOF'
import json, subprocess, sys
W, LE = sys.argv[1:3]
reg = json.load(open(f'{W}/matched/regions_matched.json'))
for i, (r0, r1, c0, c1) in enumerate(reg['letters_raw']):
    m = 60
    subprocess.run([sys.executable, f'{LE}/display.py', '--render', f'{W}/q343_letters.zarr', '--ens-dir', f'{W}/ens',
                    '--name', 'q343', '--ink', 'dmean', '--ct-d', '1', '--crop', f'{max(0, r0 - m)}:{r1 + m},{max(0, c0 - m)}:{c1 + m}',
                    '--out', f'{W}/letters/letter_{i:02d}.png'], check=True)
print('letter crops in', f'{W}/letters')
PYEOF
}
step crops 0.05 letter_crops || status "letter crops failed (non-fatal)"
VERD=$("$PY" -c "import json; print(json.load(open('$W/matched/PASS.json'))['verdict'])" 2>/dev/null || echo unknown)
status "JOB 2 done: STAGE 2 = $VERD (pre-registered matched-null rule on our surface). Observer: read $W/strips and $W/letters, record in $W/observer.txt. Next: bash $CLOUD/jobs/job3_quiet.sh"
