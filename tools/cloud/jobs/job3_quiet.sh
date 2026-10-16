#!/bin/bash
# JOB 3 - the screened scrolls of screen_scrolls.tsv, one band each (bands, fit tags: that file), in SCROLLS order:
#   inputs (uploaded or rebuilt) -> tracks fit (10k steps, dt from 7000, F10; skipped when a final
#   checkpoint is present, e.g. an uploaded one) -> official flatten -> band_screen plan -> whole-band screen
#   (dense_native, d -2/0/+2, reading side) -> top 10 % of 512 px tiles by contrast -> 3-model ensemble (ink9 s42/s43 +
#   dense_native, d -6..+6 step 2) on them -> yellow-over-CT crops + line-scale strips -> summary -> CT cache pruned.
# A band whose config z_top_to_bottom is '-' (winding sense unknown) is fitted once per sense (SENSE_MODE, default both:
# <tag>_cw and <tag>_acw, two guarded steps) and quiet/pick_sense.py picks the sense (rule: RUNBOOK.md); it refuses an
# ambiguous or failed pair, and the scroll is then skipped with the reason in the status log.
# SENSE_VALIDATE=1 first fits a band of known sense both ways (SENSE_VALIDATE_SETS, default PHerc0343x), requires pick_sense.py
# to choose the catalogue sense, and stops the job (exit 7) before any unknown-sense scroll if it does not.
# Resumable at every level; the cost guard runs before every step and every strip; a BUDGET STOP/HOLD ends the job.
JOB=job3
. "$(dirname "$0")/lib.sh"
own_group "$0" "$@"
if [ "$(cat "$STATE/STOP_RULE" 2>/dev/null)" != PASS ] && [ "${FORCE:-0}" != 1 ]; then
  status "refusing: job 1 stop rule has not passed; FORCE=1 overrides"; exit 5
fi
job_lock
"$PY" "$CLOUD/py/cost_guard.py" init --budget "$BUDGET_USD" --rate "$RATE_USD_H"
start_watchdog
: "${SCROLLS:=$SCREEN_DEFAULT}"      # default: the 'screen' rows of screen_scrolls.tsv, in file order
: "${STRIP_COLS:=16384}" "${TILE:=512}" "${TOP_FRAC:=0.10}" "${ENS_MODELS:=ink9_s42,ink9_s43,dnative}"
: "${SCREEN_DEPTHS:=-2..2:2}" "${ENS_DEPTHS:=-6..6:2}" "${PIPELINE:=1}"
[ "$OVERLAP" = 1 ] && export POD_ALLOW_FIT_OVERLAP=1
BS="$PY $CLOUD/py/band_screen.py"
if [ "$DRY" = 1 ]; then export BS_RENDER_CMD="$MOCK render" BS_SWEEP_CMD="$MOCK sweep"; fi
proj() { "$PY" -c "import json,sys; print(json.load(open('$STATE/timing.json'))['projections_GUESS']['job3_per_scroll'][sys.argv[1]])" "$1" 2>/dev/null || echo "$2"; }
status "JOB 3 start: $SCROLLS (budget guard on; DRY_RUN=$DRY; OVERLAP=$OVERLAP)"

stop_if_budget() { local rc=$1; if [ "$rc" -eq 3 ] || [ "$rc" -eq 4 ]; then status "JOB 3 stopped by the cost guard (rc=$rc)"; exit "$rc"; fi; }

prefit_next() {  # OVERLAP=1: start the next scroll's fit now (it shares the 24 GB card with this scroll's sweeps)
  local nxt=$1 tag z0 z1
  [ "$OVERLAP" = 1 ] && [ -n "$nxt" ] || return 0
  read -r tag z0 z1 <<< "${FJ[$nxt]}"
  sense_unknown "$nxt" && { status "OVERLAP: $nxt needs a fit per winding sense; no pre-fit"; return 0; }
  bash "$CLOUD/jobs/inputs.sh" status | grep -q "^$nxt .*ready=YES" || { status "OVERLAP: $nxt inputs not ready; no pre-fit"; return 0; }
  pgrep -f fit_spiral.py > /dev/null && return 0
  status "OVERLAP: pre-starting the $nxt fit ($tag) in the background"
  ( GPU_START_MAX_MB=24000 fit_tag "$tag" "$DATA/spiral/$nxt" "$z0" "$z1" 10000 "$TRACK_START" ) \
      >> "$STATE/logs/job3.${nxt}_prefit.log" 2>&1 &
}

summary() {  # S W R
  "$PY" - "$1" "$2" "$3" "$STATE" <<'PYEOF'
import json, shutil, sys, glob
from pathlib import Path
S, W, R, ST = sys.argv[1:5]; W, R = Path(W), Path(R); R.mkdir(parents=True, exist_ok=True)
plan = json.loads((W / 'plan.json').read_text()); sel = json.loads((W / 'selection.json').read_text())
idx = json.loads((R / 'index.json').read_text()) if (R / 'index.json').exists() else []
scr = [json.loads(p.read_text()) for p in sorted(W.glob('strips/*/screen.done'))]
for f in ('plan.json', 'tiles.csv', 'selection.json', 'overview_scores.png'):
    if (W / f).exists():
        shutil.copy2(W / f, R / f)
spm = [s['s_per_valid_mpx_depth'] for s in scr if s.get('valid_px', 0) > 1e5]
out = dict(scroll=S, tifxyz=plan['tifxyz'], valid_mpx=plan['valid_mpx_est'], area_cm2=plan.get('area_cm2_meta'),
           normal_sign=plan['normal_sign'], outward_fraction=plan['outward_fraction'], ct_gb_est=plan['ct_gb_est'],
           strips=len(plan['strips']), screened=len(scr), screen_sweep_s=round(sum(s['sweep_s'] for s in scr)),
           screen_render_s=round(sum(s['render_s'] for s in scr)),
           screen_s_per_valid_mpx_depth_median=sorted(spm)[len(spm) // 2] if spm else None,
           n_valid_tiles=sel['n_valid_tiles'], n_selected=sel['n_selected'], selection_rule=sel['rule'],
           clusters=len(idx), top_clusters=idx[:10],
           reading_note='orientation not established for this surface: if letters look mirrored, flip left-right')
(R / 'summary.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'top_clusters'}))
PYEOF
}

est1500() {  # fit hours of a 1500-slice band: the timing.json projection of job 2, else $1
  "$PY" -c "import json; print(round(json.load(open('$STATE/timing.json'))['projections_GUESS']['job2_stage2_343']['fit_1500_slices_h'], 2))" 2>/dev/null || echo "$1"
}

validate_sense_rule() {  # known-answer check of the pick_sense.py rule on bands of known sense; exit 7 (job stops) if it fails
  local v vds vdir vz0 vz1 sn expect rc
  for v in $SENSE_VALIDATE_SETS; do
    if [ -z "${SC[$v]:-}" ] || { [ "${ZD[$v]:-}" != true ] && [ "${ZD[$v]:-}" != false ]; }; then
      status "SENSE VALIDATION FAILED: $v is not in the screening config with a known z_top_to_bottom (true / false); the unknown-sense scrolls were NOT run"; exit 7
    fi
    read -r vz0 vz1 <<< "${BAND[$v]}"; vds=$DATA/spiral/$v; vdir=$QW/validate/$v
    status "== SENSE VALIDATION on $v (z $vz0-$vz1, catalogue z_top_to_bottom=${ZD[$v]}): fit both senses, the rule must choose the catalogue sense"
    step "val_${v}_inputs" 1.0 bash "$CLOUD/jobs/inputs.sh" ensure "$v"; rc=$?; stop_if_budget $rc
    [ $rc -eq 0 ] || { status "SENSE VALIDATION FAILED: inputs of $v are not ready; the unknown-sense scrolls were NOT run"; exit 7; }
    step "val_${v}_variants" 0.01 "$PY" "$LQ/sense_validate.py" variants --dataset "$vds" --out "$vdir"; rc=$?
    [ $rc -eq 0 ] || { status "SENSE VALIDATION FAILED: no validation specs for $v ($(tail -1 "$STATE/logs/job3.val_${v}_variants.log" | cut -c1-200)); the unknown-sense scrolls were NOT run"; exit 7; }
    expect=$(cat "$vdir/expected_sense")
    for sn in cw acw; do   # two guarded fits, like an unknown-sense scroll
      step "val_${v}_fit_${sn}" "$(est1500 1.5)" fit_tag "val_${v}_${sn}" "$vds" "$vz0" "$vz1" 10000 "$TRACK_START" "" "$vdir/spiral-scroll.${sn^^}.json"
      rc=$?; stop_if_budget $rc
      [ $rc -eq 0 ] || { status "SENSE VALIDATION FAILED: the $sn fit on $v failed; the unknown-sense scrolls were NOT run"; exit 7; }
    done
    step "val_${v}_decide" 0.01 pick_sense "val_$v" "$vds" 10000 --expect "$expect" --no-canonical --out "$vdir/sense_validation.json"; rc=$?
    stop_if_budget $rc
    if [ $rc -ne 0 ]; then
      status "SENSE VALIDATION FAILED on $v (rc=$rc): $(tail -1 "$STATE/logs/job3.val_${v}_decide.log" | cut -c1-300). The sense rule cannot be trusted, so the unknown-sense scrolls were NOT run (details: $vdir/sense_validation.json; SENSE_MODE=cw|acw forces a sense, SENSE_VALIDATE=0 skips this check)"
      exit 7
    fi
    status "$v: $(tail -1 "$STATE/logs/job3.val_${v}_decide.log" | cut -c1-300)"
  done
}

: "${SENSE_VALIDATE:=0}" "${SENSE_VALIDATE_SETS:=PHerc0343x}"
if [ "$SENSE_VALIDATE" = 1 ]; then
  unk=0; for _s in $SCROLLS; do sense_unknown "$_s" && unk=1; done
  if [ "$unk" = 1 ] && [ "$SENSE_MODE" = both ]; then validate_sense_rule
  else status "SENSE VALIDATION skipped: it needs SENSE_MODE=both and at least one unknown-sense scroll in SCROLLS"; fi
fi

set -- $SCROLLS
while [ $# -gt 0 ]; do
  S=$1; shift; NEXT=${1:-}
  read -r TAG Z0 Z1 <<< "${FJ[$S]}"
  DS=$DATA/spiral/$S; W=$WORK/cloud/quiet/$S; R=$RESULTS/$S; FLAT=$(flat_path "$TAG")
  status "== $S ($TAG, z $Z0-$Z1)"
  step "${S}_inputs" 0.5 bash "$CLOUD/jobs/inputs.sh" ensure "$S"; rc=$?; stop_if_budget $rc
  [ $rc -eq 0 ] || { status "$S: inputs not ready; skipping this scroll"; continue; }
  if sense_unknown "$S"; then
    rc=0
    for sn in $(sense_list); do   # one guarded fit per sense, each under its spec variant, into the run tag <tag>_<sense>
      step "${S}_fit_${sn}" "$(proj fit_h 1.0)" fit_tag "${TAG}_${sn}" "$DS" "$Z0" "$Z1" 10000 "$TRACK_START" "" "$DS/spiral-scroll.${sn^^}.json"
      rc=$?; stop_if_budget $rc
      [ $rc -eq 0 ] || { status "$S: fit ($sn) failed; skipping"; break; }
    done
    [ $rc -eq 0 ] || continue
    step "${S}_sense" 0.01 pick_sense "$TAG" "$DS" 10000; rc=$?; stop_if_budget $rc
    if [ $rc -ne 0 ]; then status "$S: SENSE DECISION FAILED (rc=$rc): $(tail -1 "$STATE/logs/job3.${S}_sense.log" | cut -c1-300); skipping"; continue; fi
    status "$S: $(tail -1 "$STATE/logs/job3.${S}_sense.log" | cut -c1-300)"
  else
    step "${S}_fit" "$(proj fit_h 1.0)" fit_tag "$TAG" "$DS" "$Z0" "$Z1" 10000 "$TRACK_START"; rc=$?; stop_if_budget $rc
    [ $rc -eq 0 ] || { status "$S: fit failed; skipping"; continue; }
  fi
  step "${S}_flatten" 0.15 flatten_tag "$TAG" "$DS" 10000 "${VX[$S]:-8.64}"; rc=$?; stop_if_budget $rc
  [ $rc -eq 0 ] || { status "$S: flatten failed; skipping"; continue; }
  if [ "${SNAP:-0}" = 1 ]; then
    step "${S}_snap" 0.3 snap_flat "${SC[$S]}" "$TAG"; rc=$?; stop_if_budget $rc   # scroll name, not the dataset name
    if [ $rc -eq 0 ]; then FLAT=$(snap_path "$TAG"); status "$S: screening the snapped mesh $FLAT"
    else status "$S: snap failed (rc=$rc); screening the unsnapped mesh"; fi
  fi
  prefit_next "$NEXT"
  step "${S}_plan" 0.05 $BS plan --scroll "${SC[$S]}" --tifxyz "$FLAT" --out "$W" --strip-cols "$STRIP_COLS" --tile "$TILE"; rc=$?
  stop_if_budget $rc
  [ $rc -eq 0 ] || { status "$S: plan failed (rc=$rc; 3 = ambiguous normal sign); skipping"; continue; }
  "$PY" -c "import json; p=json.load(open('$W/plan.json')); print(f\"plan: {p['valid_mpx_est']} valid Mpx, {len(p['strips'])} strips, CT ~{p['ct_gb_est']} GB, normals {p['normal_sign']} (outward {p['outward_fraction']:.3f})\")" | while read -r l; do status "$S $l"; done
  step "${S}_screen" 0.05 $BS screen --plan "$W/plan.json" --device "$DEVICE" --batch "$BATCH" --procs "$RENDER_PROCS" \
       --depths="$SCREEN_DEPTHS" --pipeline "$PIPELINE"; rc=$?; stop_if_budget $rc
  [ $rc -eq 0 ] || { status "$S: screen failed; skipping"; continue; }
  step "${S}_select" 0.02 $BS select --plan "$W/plan.json" --depths="$SCREEN_DEPTHS" --top-frac "$TOP_FRAC"; rc=$?
  stop_if_budget $rc; [ $rc -eq 0 ] || { status "$S: select failed; skipping"; continue; }
  step "${S}_ens" 0.05 $BS ens --plan "$W/plan.json" --models "$ENS_MODELS" --depths="$ENS_DEPTHS" --device "$DEVICE" \
       --batch "$BATCH" --procs "$RENDER_PROCS" --results "$R"; rc=$?; stop_if_budget $rc
  [ $rc -eq 0 ] || { status "$S: ensemble failed; skipping"; continue; }
  step "${S}_summary" 0.01 summary "$S" "$W" "$R"
  tail -1 "$STATE/logs/job3.${S}_summary.log" | cut -c1-400 | while read -r l; do status "$S summary: $l"; done
  step "${S}_prune" 0.01 prune_ct "${SC[$S]}"
done
status "JOB 3 finished the scroll list. Sync results back (pc/sync_back.sh), then stop/terminate the pod."
