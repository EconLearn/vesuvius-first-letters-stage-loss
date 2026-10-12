#!/bin/bash
# Local dry run of the cloud kit: CPU only, no GPU, no big downloads, everything under DRY_BASE (< 200 MB, deleted at the
# end; logs kept in DRY_LOGS). Steps:
#   syntax of every script -> pack_code.sh (< 50 MB) -> unpack into a fake code dir -> bootstrap.sh --dry-run
#   [--check-locks: PyPI metadata only; DRY_CHECK_LOCKS=0 skips it] -> jobs 1, 2, 3 with DRY_RUN=1 mocks (real helper code
#   on mock outputs) -> resume (re-run = all skipped) -> budget hold -> stop-rule FAIL path -> stage-2 NOT TESTABLE path ->
#   sync_back --dry-run (local) -> size check.
# Unknown winding sense (quiet/pick_sense.py): the test config gets two mock scrolls with z_top_to_bottom = '-'; one fit per
# sense with a clearly better CW fit, an ambiguous pair (must be refused), SENSE_MODE=cw forcing, and direct pick_sense.py
# cases (ACW wins, 2 % boundary, non-finite loss, crashed fit). Mock scroll A starts with NO descriptors (built from the m7
# prediction by inputs.sh) and has a 9.362 um volume (config-driven paths and voxel size); SENSE_VALIDATE=1 is exercised with a
# passing and a failing known-answer check (the failing one must stop the job before any unknown-sense scroll).
# env: POD_PY = python with the repository's requirements (default $VC_ROOT/.venv/bin/python), DRY_BASE, DRY_LOGS, KEEP=1.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1                  # keep the repository free of __pycache__ written by the test
C=$(cd "$(dirname "$0")/.." && pwd)                  # tools/cloud
BASE=${DRY_BASE:-${TMPDIR:-/tmp}/cloud_dry}
LOGD=${DRY_LOGS:-${TMPDIR:-/tmp}/cloud_dry_logs}
PYPC=${POD_PY:-${VC_ROOT:-$HOME/vesuvius}/.venv/bin/python}
rm -rf "$BASE" "$LOGD"; mkdir -p "$BASE" "$LOGD"
pass=0; fail=0
ok() { echo "PASS $1"; pass=$((pass + 1)); }
ko() { echo "FAIL $1 ($2)"; fail=$((fail + 1)); }
expect() {  # expect NAME RC CMD... : run, log, compare the exit code
  local name=$1 want=$2; shift 2
  "$@" > "$LOGD/$name.log" 2>&1; local rc=$?
  if [ "$rc" = "$want" ]; then ok "$name (rc=$rc)"; else ko "$name" "rc=$rc, expected $want; log $LOGD/$name.log"; fi
}
has() { [ -e "$2" ] && ok "$1" || ko "$1" "missing $2"; }
# test config = the screening config (SCREEN_CONFIG or the shipped one) + two mock scrolls whose winding sense is unknown ('-')
SRC_CFG=${SCREEN_CONFIG:-$C/screen_scrolls.tsv}; export SCREEN_CONFIG=$BASE/dry_screen.tsv
S_UA=MockScrollA; S_UB=MockScrollB
{ cat "$SRC_CFG"
  printf '%s\t%s\tq_mock_a\t1000\t2000\t0\tscreen\t20990101000000-9.362um-1.2m-116keV-masked.zarr\t4000\t3000\t3000\t-\t9.362\n' $S_UA $S_UA
  printf '%s\t%s\tq_mock_b\t1000\t2000\t0\tscreen\t20990202000000-8.640um-1.2m-116keV-masked.zarr\t4000\t3000\t3000\t-\t8.64\n' $S_UB $S_UB
} > "$SCREEN_CONFIG"
# scrolls used by the checks, taken from whatever the screening config holds (rows of group 'screen' with a known sense, in file
# order): first and last row for the main run, the first two rows for the OVERLAP check (skipped with a single row)
mapfile -t SCR < <(awk -F'\t' '$1 !~ /^#/ && $1 != "dataset" && $7 == "screen" && $12 != "-" {print $1}' "$SCREEN_CONFIG")
NS=${#SCR[@]}; [ "$NS" -ge 1 ] || { echo "no 'screen' rows in the screening config"; exit 2; }
SA=${SCR[0]}; SB=${SCR[NS-1]}; S_BUD=${SCR[0]}; S_O1=${SCR[0]}; S_O2=${SCR[1]:-}

echo "== 1 syntax"
expect syntax_bash 0 bash -c "for f in $C/*.sh $C/jobs/*.sh $C/pc/*.sh $C/tests/*.sh $C/gpu/*.sh; do bash -n \$f || { echo BAD \$f; exit 1; }; done"
expect syntax_py 0 env PYTHONPYCACHEPREFIX="$BASE/pycache" "$PYPC" -m py_compile $C/py/*.py $C/pod/*.py $C/tests/*.py $C/quiet/*.py $C/read/*.py $C/gpu/*.py
for t in quiet_super build_normals_band tracks_subset umb_auto umb_smooth pick_sense sense_validate; do expect help_$t 0 "$PYPC" $C/quiet/$t.py --help; done
expect help_band_screen 0 "$PYPC" $C/py/band_screen.py --help
expect help_cost_guard 0 "$PYPC" $C/py/cost_guard.py --help
expect help_timing 0 "$PYPC" $C/py/timing_report.py --help

echo "== 2 pack"
expect pack 0 bash $C/pack_code.sh "$BASE/dist"
SZ=$(stat -c %s "$BASE/dist/vesuvius_code.tgz" 2>/dev/null || echo 0)
[ "$SZ" -gt 0 ] && [ "$SZ" -lt $((50 * 1024 * 1024)) ] && ok "pack_size $((SZ / 1024)) KiB < 50 MB" || ko pack_size "$SZ bytes"
for bad in '\.zarr/' '\.npy$' '\.tif$' '\.dbm' '\.ckpt$' '\.pth$' '\.safetensors$' '\.so$' 'data/s3/'; do
  if tar -tzf "$BASE/dist/vesuvius_code.tgz" | grep -qE "$bad"; then ko "pack_excludes $bad" "found in tarball"; fi
done
ok "pack_excludes (no data/weights/renders/.so in the tarball)"

echo "== 3 unpack into a fake code dir + bootstrap --dry-run"
FAKE_CODE=$BASE/code; FAKE=$BASE/vc_root; mkdir -p "$FAKE_CODE" "$FAKE"; tar -xzf "$BASE/dist/vesuvius_code.tgz" -C "$FAKE_CODE"
LOCKF=--check-locks; [ "${DRY_CHECK_LOCKS:-1}" = 0 ] && LOCKF=""
expect bootstrap_dry 0 env VC_ROOT="$FAKE" POD_STATE="$BASE/bstate" bash "$FAKE_CODE/tools/cloud/bootstrap.sh" --dry-run $LOCKF

echo "== 4 jobs with mocks"
export DRY_RUN=1 VC_ROOT=$FAKE POD_STATE=$FAKE/cloud_state POD_PY=$PYPC POD_PYX=$PYPC CLOUD=$FAKE_CODE/tools/cloud
export BS_EDGE_ERODE=${BS_EDGE_ERODE:-0}   # the mock canvases are far smaller than the default 128 px edge erosion of band_screen select
mkdir -p "$POD_STATE" "$FAKE/data/spiral/PHerc0343"
# input of job 2's umbilicus union (a real run has the dataset descriptor here): two mock control points
printf '{"control_points": [{"z": 8100, "x": 4300.0, "y": 4300.0}, {"z": 9900, "x": 4310.0, "y": 4290.0}]}\n' > "$FAKE/data/spiral/PHerc0343/umbilicus.json"
cat > "$POD_STATE/pod.env" <<EOF
TIMING_ROWS=0:240
TIMING_COLS=0:320
STRIP_COLS=1024
TILE=128
SCROLLS="$SA $SB"
DEVICE=cpu
RENDER_PROCS=1
PRUNE_CT=1
EOF
"$PYPC" "$CLOUD/py/cost_guard.py" init --t0 "$(date +%s)" --budget 25 --rate 0.34 > /dev/null
J=$CLOUD/jobs
expect job1 0 bash "$J/job1_timing.sh"
has timing_json "$POD_STATE/timing.json"
grep -q PASS "$POD_STATE/STOP_RULE" 2>/dev/null && ok stop_rule_pass_recorded || ko stop_rule_pass_recorded "no PASS"
sleep 3   # background input builders (mock) finish in seconds
expect job2 0 bash "$J/job2_stage2_343.sh"
has stage2_pass_json "$FAKE/work/ensemble/stage2x/matched/PASS.json"
"$PYPC" -c "import json; v=json.load(open('$FAKE/work/ensemble/stage2x/matched/PASS.json')); print(v['verdict']); assert v['verdict']=='PASS'" \
   > "$LOGD/stage2_verdict.log" 2>&1 && ok "stage2 matched-null scorer gives PASS on mock letters with ink" || ko stage2_verdict "see log"
expect job3 0 bash "$J/job3_quiet.sh"
for S in $SA $SB; do
  has "results_$S" "$FAKE/out/cloud/$S/summary.json"
  ls "$FAKE"/out/cloud/$S/strips/*_strips/strip_00_r0.png > /dev/null 2>&1 && ok "strips_png_$S" || ko "strips_png_$S" "no strip PNGs"
done
has inputs_markers "$FAKE/data/spiral/$SB/.trk_done"

echo "== 4a volume id and voxel size come from the config (real m7normals / umb_smooth / volumes code)"
"$PYPC" - "$C" "$S_UA" > "$LOGD/config_driven_paths.log" 2>&1 <<'PYEOF' && ok "9.362 um volume: CT path, voxel size and registry follow the config, no 8.64 um pattern" || ko config_driven_paths "see log"
import sys
c, s = sys.argv[1:3]
sys.path.insert(0, c + '/quiet'); sys.path.insert(0, c + '/py')
import m7normals as M, umb_smooth as U, volumes
vid = '20990101000000-9.362um-1.2m-116keV-masked.zarr'
assert M.VOLID[s] == vid and M.VOL[s] == '20990101000000' and M.VOXEL[s] == 9.362 and M.ZDIR[s] is None and M.SHAPE[s] == [4000, 3000, 3000]
ct = U.CT.format(s=s, vid=M.VOLID[s])
assert ct == f'vesuvius-challenge-open-data/{s}/volumes/{vid}' and '8.640' not in ct and '8.64um' not in ct, ct
C = volumes.register()
assert C.VOLUMES[s]['vol_id'] == vid and C.VOLUMES[s]['um'] == 9.362 and C.VOLUMES['PHerc0343']['um'] == 8.64, C.VOLUMES[s]
assert str(volumes.mirror_dir(s)).endswith(f'/s3/{s}/volumes/{vid}/0')
print(ct, M.VOXEL[s])
PYEOF

echo "== 4b unknown winding sense: descriptors built, one guarded fit per sense, loss comparison, refusal of an ambiguous pair, forcing"
MAIN_STATE=$POD_STATE
export POD_STATE=$BASE/state_sense; mkdir -p "$POD_STATE"; cp "$MAIN_STATE/pod.env" "$POD_STATE/"; echo PASS > "$POD_STATE/STOP_RULE"
"$PYPC" "$CLOUD/py/cost_guard.py" init --t0 "$(date +%s)" > /dev/null
export MOCK_LOSS_MAP="q_mock_a=270/410;q_mock_b=300/303"          # A: CW clearly better; B: 1 % apart (ambiguous)
echo "SCROLLS=\"$S_UA\"" >> "$POD_STATE/pod.env"
SD=$FAKE/data/spiral/$S_UA
rm -rf "$SD" "$FAKE/data/spiral/$S_UB"       # mock scrolls A and B start with no descriptors at all (job 1 had pre-built them)
[ ! -e "$SD/umbilicus.json" ] && [ ! -e "$SD/spiral-scroll.json" ] && ok "mock scroll starts without any descriptor" || ko sense_no_descriptors_start "still present"
expect job3_sense_clear 0 bash "$J/job3_quiet.sh"
[ -s "$SD/umbilicus.json" ] && [ -e "$SD/spiral-scroll.CW.json" ] && [ -e "$SD/spiral-scroll.ACW.json" ] && [ -e "$SD/.desc_done" ] \
  && [ -e "$FAKE/out/cloud/$S_UA/descriptors/umbilicus_qa.png" ] && [ -e "$FAKE/out/cloud/$S_UA/descriptors/umbilicus_qa_core_zoom.png" ] \
  && grep -q "$S_UA: descriptors ready" "$POD_STATE/STATUS.log" \
  && ok "descriptors built on the pod (umbilicus, spec + variants, QA images in the results dir, status line)" || ko sense_descriptors_built "see $POD_STATE/STATUS.log"
"$PYPC" -c "import json; d=json.load(open('$SD/spiral-scroll.nosense.json')); assert d['voxel_size_um']==9.362 and d['z_direction_is_top_to_bottom'] is None, d; p=json.load(open('$FAKE/work/cloud/quiet/$S_UA/plan.json')); assert p['voxel_um']==9.362, p['voxel_um']; m=json.load(open('$FAKE/work/read/q_mock_a/flat/q_mock_a.tifxyz/meta.json')); assert m['voxel_um']==9.362, m" \
  > "$LOGD/sense_voxel_plumbing.log" 2>&1 && ok "voxel size 9.362 reaches the spec, the screening plan and the flatten" || ko sense_voxel_plumbing "see log"
[ "$(grep -c " START ${S_UA}_fit_" "$POD_STATE/STATUS.log")" = 2 ] && [ "$(grep -c "COST job3:${S_UA}_fit_" "$POD_STATE/STATUS.log")" = 2 ] \
  && ok "two fits (cw, acw), each behind its own cost-guard check" || ko sense_two_guarded_fits "see $POD_STATE/STATUS.log"
"$PYPC" - "$SD/sense_decision.json" > "$LOGD/sense_decision_check.log" 2>&1 <<'PYEOF' && ok "decision JSON: winner CW, both losses, rule recorded" || ko sense_decision_json "see log"
import json, sys
d = json.load(open(sys.argv[1]))
assert d['decision'] == 'winner' and d['winner'] == 'CW' and d['mode'] == 'both', d
assert d['senses']['CW']['mean_loss'] < d['senses']['ACW']['mean_loss'] and d['rel_diff'] > 0.4 and d['rule'], d
print(d['senses']['CW']['mean_loss'], d['senses']['ACW']['mean_loss'], d['rel_diff'])
PYEOF
"$PYPC" - "$FAKE/work/quiet/runs/q_mock_a" "$SD/spiral-scroll.json" > "$LOGD/sense_canonical_check.log" 2>&1 <<'PYEOF' && ok "winner is the canonical run (checkpoint + spiral-scroll.json), sense-less spec kept as .nosense" || ko sense_canonical "see log"
import json, os, sys, torch
run, spec = sys.argv[1:3]
ck = os.path.join(run, 'checkpoint_fitted.ckpt')
assert os.path.islink(ck) and 'q_mock_a_cw' in os.readlink(ck), (ck, os.readlink(ck))
assert int(torch.load(ck, map_location='cpu', weights_only=False)['completed_iterations']) == 10000
assert json.load(open(spec))['spiral_outward_sense'] == 'CW'
assert json.load(open(os.path.join(os.path.dirname(spec), 'spiral-scroll.nosense.json'))).get('spiral_outward_sense') is None
PYEOF
grep -q "SENSE DECISION q_mock_a: winner=CW" "$POD_STATE/STATUS.log" && ok "decision in the job status log" || ko sense_status_log "no SENSE DECISION line"
has sense_scroll_screened "$FAKE/out/cloud/$S_UA/summary.json"
NS0=$(grep -c ' START ' "$POD_STATE/STATUS.log")
expect job3_sense_resume 0 bash "$J/job3_quiet.sh"
[ "$NS0" = "$(grep -c ' START ' "$POD_STATE/STATUS.log")" ] && ok "sense flow resumes: nothing restarted" || ko sense_resume "START lines changed"
sed -i '/^SCROLLS=/d' "$POD_STATE/pod.env"; echo "SCROLLS=\"$S_UB\"" >> "$POD_STATE/pod.env"
expect job3_sense_ambiguous 0 bash "$J/job3_quiet.sh"
grep -q "SENSE DECISION FAILED.*ambiguous" "$POD_STATE/STATUS.log" && ok "ambiguous pair refused with a clear message" || ko sense_ambiguous_msg "no FAILED/ambiguous line"
[ ! -e "$FAKE/out/cloud/$S_UB/summary.json" ] && [ ! -e "$FAKE/work/quiet/runs/q_mock_b/checkpoint_fitted.ckpt" ] \
  && ok "refused scroll: not screened, no canonical run" || ko sense_ambiguous_effect "summary or canonical run exists"
"$PYPC" -c "import json,sys; d=json.load(open('$FAKE/data/spiral/$S_UB/sense_decision.json')); assert d['decision']=='refused' and 'ambiguous' in d['reason'], d" \
  > "$LOGD/sense_refused_json.log" 2>&1 && ok "refusal recorded in sense_decision.json" || ko sense_refused_json "see log"
echo "SENSE_MODE=cw" >> "$POD_STATE/pod.env"
expect job3_sense_forced 0 bash "$J/job3_quiet.sh"
has sense_forced_screened "$FAKE/out/cloud/$S_UB/summary.json"
"$PYPC" -c "import json; d=json.load(open('$FAKE/data/spiral/$S_UB/sense_decision.json')); assert d['decision']=='forced' and d['winner']=='CW', d" \
  > "$LOGD/sense_forced_json.log" 2>&1 && ok "SENSE_MODE=cw forces the sense (decision forced)" || ko sense_forced_json "see log"

echo "-- descriptor build failure is reported like the other input steps, and the next run builds them"
export POD_STATE=$BASE/state_descfail; mkdir -p "$POD_STATE"; cp "$MAIN_STATE/pod.env" "$POD_STATE/"; echo PASS > "$POD_STATE/STOP_RULE"
echo "SCROLLS=\"$S_UB\"" >> "$POD_STATE/pod.env"
"$PYPC" "$CLOUD/py/cost_guard.py" init --t0 "$(date +%s)" > /dev/null
rm -rf "$FAKE/data/spiral/$S_UB"
expect job3_desc_fail 0 env MOCK_DESC_FAIL=1 bash "$J/job3_quiet.sh"
grep -q "$S_UB: descriptors FAILED" "$POD_STATE/STATUS.log" && grep -q "$S_UB: inputs not ready; skipping this scroll" "$POD_STATE/STATUS.log" \
  && [ ! -e "$FAKE/data/spiral/$S_UB/.desc_done" ] && ok "failed descriptor build: status line, scroll skipped, no marker" || ko desc_fail_status "see $POD_STATE/STATUS.log"
expect job3_desc_resume 0 bash "$J/job3_quiet.sh"
has desc_resume_built "$FAKE/data/spiral/$S_UB/.desc_done"

echo "== 4c SENSE_VALIDATE=1: known-answer check of the rule before any unknown-sense scroll (fails first, then passes)"
VAL=PHerc0343x
val_state() {  # val_state NAME SCROLL : fresh job state with SENSE_VALIDATE=1 and no earlier validation fits
  export POD_STATE=$BASE/$1; mkdir -p "$POD_STATE"; cp "$MAIN_STATE/pod.env" "$POD_STATE/"; echo PASS > "$POD_STATE/STOP_RULE"
  printf 'SCROLLS="%s"\nSENSE_VALIDATE=1\n' "$2" >> "$POD_STATE/pod.env"
  "$PYPC" "$CLOUD/py/cost_guard.py" init --t0 "$(date +%s)" > /dev/null
  rm -rf "$FAKE"/work/quiet/runs/val_${VAL}_* "$FAKE"/work/quiet/logs/fit_val_${VAL}_* "$FAKE/work/quiet/validate/$VAL"
}
export MOCK_LOSS_MAP="val_$VAL=410/270;q_mock_a=270/410;q_mock_b=300/303"       # the wrong sense (ACW) fits the known band better
val_state state_valfail "$S_UB"
expect job3_validate_fail 7 bash "$J/job3_quiet.sh"
grep -q "SENSE VALIDATION FAILED on $VAL.*chose ACW but the catalogue sense is CW.*NOT run" "$POD_STATE/STATUS.log" \
  && ok "validation failure: clear message, job stops" || ko validate_fail_msg "see $POD_STATE/STATUS.log"
[ "$(grep -c "COST job3:val_${VAL}_fit_" "$POD_STATE/STATUS.log")" = 2 ] && ! grep -q " START ${S_UB}_" "$POD_STATE/STATUS.log" \
  && ok "validation fits are cost-guarded; no unknown-sense scroll was started" || ko validate_fail_effect "see $POD_STATE/STATUS.log"
"$PYPC" -c "import json; d=json.load(open('$FAKE/work/quiet/validate/$VAL/sense_validation.json')); assert d['decision']=='validation_failed' and d['expect']=='CW' and d['winner']=='ACW', d" \
  > "$LOGD/validate_fail_json.log" 2>&1 && ok "failed validation recorded (expected CW, rule chose ACW)" || ko validate_fail_json "see log"
export MOCK_LOSS_MAP="q_mock_a=270/410;q_mock_b=300/303"                         # default levels for val_*: CW clearly better = catalogue sense
val_state state_valok "$S_UA"
expect job3_validate_ok 0 bash "$J/job3_quiet.sh"
grep -q "SENSE VALIDATION PASSED val_$VAL: the rule chose CW = the catalogue sense" "$POD_STATE/STATUS.log" \
  && ok "validation passes: rule chose the catalogue sense CW" || ko validate_ok_msg "see $POD_STATE/STATUS.log"
[ "$(grep -c "COST job3:val_${VAL}_fit_" "$POD_STATE/STATUS.log")" = 2 ] && has validate_ok_screened "$FAKE/out/cloud/$S_UA/summary.json"
"$PYPC" -c "import json; d=json.load(open('$FAKE/work/quiet/validate/$VAL/sense_validation.json')); assert d['validated'] is True and d['winner']=='CW' and 'canonical' not in d, d; assert not __import__('os').path.exists('$FAKE/work/quiet/runs/val_$VAL/checkpoint_fitted.ckpt')" \
  > "$LOGD/validate_ok_json.log" 2>&1 && ok "passed validation recorded; no canonical run or spec touched" || ko validate_ok_json "see log"
export POD_STATE=$MAIN_STATE; unset MOCK_LOSS_MAP

echo "== 4d pick_sense.py rule on prepared fit pairs"
PS=$BASE/ps
ps_case() {  # ps_case NAME LOSS_MAP [cw-only] : mock fits t_cw / t_acw (loss level per sense from the map) + spec variants
  local d=$PS/$1 sn; rm -rf "$d"; mkdir -p "$d/ds"
  printf '{"name": "x", "z_direction_is_top_to_bottom": null}\n' > "$d/ds/spiral-scroll.json"
  for sn in CW ACW; do printf '{"name": "x", "spiral_outward_sense": "%s"}\n' $sn > "$d/ds/spiral-scroll.$sn.json"; done
  for sn in cw acw; do
    [ "${3:-}" = cw-only ] && [ $sn = acw ] && continue
    MOCK_LOSS_MAP="$2" QUIET_WORK="$d/qw" "$PYPC" "$C/tests/mock_tools.py" fit t_$sn 10000 7000 "$d/ds/spiral-scroll.${sn^^}.json" > /dev/null
  done
}
ps_pick() {  # ps_pick NAME WANT_RC [MODE]
  expect "sense_$1" "$2" env QUIET_WORK="$PS/$1/qw" "$PYPC" "$C/quiet/pick_sense.py" --tag t --dataset "$PS/$1/ds" --steps 10000 --mode "${3:-both}"
}
ps_case acw_wins "t=400/300";            ps_pick acw_wins 0
grep -q "winner=ACW" "$LOGD/sense_acw_wins.log" && grep -q '"spiral_outward_sense": "ACW"' "$PS/acw_wins/ds/spiral-scroll.json" && ok "lower ACW loss wins; its spec becomes spiral-scroll.json" || ko sense_acw_wins_effect "see log"
ps_case below_2pct "t=300/305.7";        ps_pick below_2pct 6
grep -q "ambiguous" "$LOGD/sense_below_2pct.log" && ok "1.9 % apart is ambiguous" || ko sense_below_2pct_msg "see log"
ps_case above_2pct "t=300/306.3";        ps_pick above_2pct 0
ps_case nonfinite "t=300/nan";           ps_pick nonfinite 6
grep -q "non-finite" "$LOGD/sense_nonfinite.log" && ok "non-finite loss is refused" || ko sense_nonfinite_msg "see log"
ps_case crashed "t=300/400" cw-only;     ps_pick crashed 6
grep -q "ACW fit crashed or is incomplete" "$LOGD/sense_crashed.log" && ok "a missing fit is reported as crashed or incomplete" || ko sense_crashed_msg "see log"
ps_case forced_acw "t=300/305" ;         ps_pick forced_acw 0 acw
grep -q "forced" "$LOGD/sense_forced_acw.log" && ok "forced mode takes the requested sense without comparing" || ko sense_forced_acw_msg "see log"

echo "== 5 resume, budget, stop-rule and coverage paths"
N0=$(grep -c ' START ' "$POD_STATE/STATUS.log")
expect job3_resume 0 bash "$J/job3_quiet.sh"
N1=$(grep -c ' START ' "$POD_STATE/STATUS.log")
[ "$N0" = "$N1" ] && ok "resume: second job3 run started no step ($N0 = $N1 START lines)" || ko resume "$N0 != $N1"
# OVERLAP=1 on a fresh job state (the steps run again; fits and flattened surfaces already exist)
if [ -n "$S_O2" ]; then
  export POD_STATE=$BASE/state_ov; mkdir -p "$POD_STATE"; cp "$FAKE/cloud_state/pod.env" "$POD_STATE/"; echo PASS > "$POD_STATE/STOP_RULE"
  printf 'SCROLLS="%s %s"\nOVERLAP=1\n' "$S_O1" "$S_O2" >> "$POD_STATE/pod.env"
  "$PYPC" "$CLOUD/py/cost_guard.py" init --t0 "$(date +%s)" > /dev/null
  expect job3_overlap 0 bash "$J/job3_quiet.sh"
  grep -q "OVERLAP: pre-starting the $S_O2 fit" "$POD_STATE/STATUS.log" && ok "OVERLAP=1 pre-starts the next scroll's fit" || ko overlap "no pre-start line"
  has overlap_results "$FAKE/out/cloud/$S_O2/summary.json"
else echo "SKIP overlap check: the screening config has a single 'screen' row"; fi
# budget stop on a fresh job state with a 0.01 USD budget
export POD_STATE=$BASE/state_bud; mkdir -p "$POD_STATE"; cp "$FAKE/cloud_state/pod.env" "$POD_STATE/"; echo PASS > "$POD_STATE/STOP_RULE"
echo "SCROLLS=\"$S_BUD\"" >> "$POD_STATE/pod.env"
"$PYPC" "$CLOUD/py/cost_guard.py" init --t0 "$(date +%s)" --budget 0.01 > /dev/null
expect job3_budget_stop 3 bash "$J/job3_quiet.sh"
grep -q 'BUDGET STOP' "$POD_STATE/STATUS.log" && ok "budget stop logged" || ko budget_log "no BUDGET STOP line"
# fresh state: slow fit -> stop rule FAIL -> job 2 refuses
export POD_STATE=$BASE/state_fail; mkdir -p "$POD_STATE"; cp "$FAKE/cloud_state/pod.env" "$POD_STATE/"
"$PYPC" "$CLOUD/py/cost_guard.py" init --t0 "$(date +%s)" > /dev/null
rm -f "$FAKE/work/quiet/logs/fit_t343x_500.log" "$FAKE/work/quiet/runs/t343x_500/checkpoint_fitted.ckpt"
expect job1_stoprule_fail 5 env MOCK_FIT_RATE=1.6,1.3 bash "$J/job1_timing.sh"
expect job2_refuses 5 bash "$J/job2_stage2_343.sh"
# fresh state: stage 2 NOT TESTABLE (our surface misses his letters)
export POD_STATE=$BASE/state_cov; mkdir -p "$POD_STATE"; cp "$FAKE/cloud_state/pod.env" "$POD_STATE/"; echo PASS > "$POD_STATE/STOP_RULE"
"$PYPC" "$CLOUD/py/cost_guard.py" init --t0 "$(date +%s)" > /dev/null
rm -rf "$FAKE/work/ensemble/stage2x"
expect job2_not_testable 10 env MOCK_STAGE2_RC=10 bash "$J/job2_stage2_343.sh"
export POD_STATE=$FAKE/cloud_state

echo "== 6 sync_back (local mode, dry) + cleanup"
expect sync_back_dry 0 env POD=local POD_ROOT_REMOTE="$FAKE" SYNC_DST="$BASE/synced" bash "$C/pc/sync_back.sh" --dry-run
expect sync_back_real 0 env POD=local POD_ROOT_REMOTE="$FAKE" SYNC_DST="$BASE/synced" bash "$C/pc/sync_back.sh"
find "$BASE/synced" -name '*.zarr' -o -name '*.npy' -o -name '*.tif' | grep -q . && ko sync_small_only "pulled big files" || ok "sync_back pulled only small files ($(du -sh "$BASE/synced" | cut -f1))"
cp "$FAKE/cloud_state/STATUS.log" "$LOGD/STATUS_dryrun.log" 2>/dev/null
cp "$BASE/state_fail/STATUS.log" "$LOGD/STATUS_stoprule_fail.log" 2>/dev/null
cp "$BASE/state_cov/STATUS.log" "$LOGD/STATUS_not_testable.log" 2>/dev/null
cp "$FAKE/cloud_state/timing.json" "$LOGD/timing_mock.json" 2>/dev/null
PEAK=$(du -sm "$BASE" | cut -f1)
[ "$PEAK" -lt 200 ] && ok "dry-run footprint ${PEAK} MB < 200 MB" || ko footprint "${PEAK} MB"
pkill -f "cost_guard.py watch" 2>/dev/null
[ "${KEEP:-0}" = 1 ] || rm -rf "$BASE"
echo "DRY RUN: $pass passed, $fail failed (logs: $LOGD)"
[ "$fail" -eq 0 ]
