#!/bin/bash
# Fit inputs per dataset (normals + tracks + crossings [+ umbilicus for PHerc0343x]) on the pod, from public data only,
# with the input-builder scripts under $QUIET_DIR (tools/cloud/quiet), or accept a dataset uploaded
# from a workstation (pc/upload_inputs.sh writes the markers).
#   inputs.sh status                      readiness of every dataset
#   inputs.sh ensure DS [DS...]           wait for an upload in progress, else build what is missing (flock per DS)
#   inputs.sh bg DS [DS...]               the same, detached, nice 19 (log: $STATE/logs/inputs_bg.log)
# Markers in the dataset dir ($VC_DATA/spiral/DS): .norm_done .desc_done .umb_done .trk_done (.upload_in_progress / .upload_done
# from the upload). Descriptors (umbilicus.json, spiral-scroll*.json) that are missing are built first, from the published
# m7 prediction (quiet/umb_auto.py, quiet/umb_smooth.py; volume id, shape, voxel size and z-direction from the screening
# config); the PHerc0343x band has its own lasagna-based route. Datasets, scrolls and z-bands = screen_scrolls.tsv (SC / BAND in lib.sh); normals band = fit band +- 200
# slices; PHerc0343x = the stage-2 band 9500-11000 of PHerc0343 with the PUBLISHED lasagna (fetch_lasagna_band.py).
JOB=inputs
. "$(dirname "$0")/lib.sh"
: "${UPLOAD_WAIT_H:=6}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

ds_dir() { echo "$DATA/spiral/$1"; }

ready() {   # the fit queue's ready() test + our markers
  local d; d=$(ds_dir "$1")
  [ -e "$d/.upload_in_progress" ] && return 1
  { [ -e "$d/.norm_done" ] || [ -e "$d/.upload_done" ]; } && [ -e "$d/.trk_done" ] || return 1
  [ "$1" = PHerc0343x ] && [ ! -e "$d/.umb_done" ] && return 1
  "$PY" - "$d" <<'PYEOF'
import json, os, sys
d = sys.argv[1]; s = json.load(open(os.path.join(d, 'spiral-scroll.json'))); p = s.get('paths', {})
ok_norm = os.path.isdir(os.path.join(d, p.get('normal_x', 'x'), '2')) and os.path.exists(os.path.join(d, 'umbilicus.json'))
trk = os.path.join(d, p.get('tracks_dbm', 'none'))
ok_trk = os.path.exists(trk) and os.path.exists(trk + '.crossings.npz')
ok_sense = s.get('z_direction_is_top_to_bottom') is not None or s.get('spiral_outward_sense') in ('CW', 'ACW')
# unknown winding sense: job 3 fits the spec variants spiral-scroll.<SENSE>.json named by SENSE_MODE (cw / acw / both)
need = {'both': ['CW', 'ACW'], 'cw': ['CW'], 'acw': ['ACW']}.get(os.environ.get('SENSE_MODE', 'both'), [])
ok_sense = ok_sense or (bool(need) and all(os.path.exists(os.path.join(d, f'spiral-scroll.{x}.json')) for x in need))
sys.exit(0 if (ok_norm and ok_trk and ok_sense) else 1)
PYEOF
}

build_normals() {  # DS
  local ds=$1 d s z0 z1; d=$(ds_dir "$ds"); s=${SC[$ds]}; read -r z0 z1 <<< "${BAND[$ds]}"
  [ -e "$d/.norm_done" ] && return 0
  mkdir -p "$d"
  if [ "$DRY" = 1 ]; then $MOCK inputs norm "$d" "$s" "$z0" "$z1" && touch "$d/.norm_done"; return $?; fi
  if [ "$ds" = PHerc0343x ]; then
    nice -n 19 "$PY" -u "$LQ/fetch_lasagna_band.py" $((z0 - 200)) $((z1 + 200)) "$d" || return 2
  else
    nice -n 19 "$PY" -u "$LQ/build_normals_band.py" "$s" $((z0 - 200)) $((z1 + 200)) "$d" --sigma-d 1.0 \
       --sigma-i 2.5 --coh-min 0.6 --no-density --workers "$INPUT_WORKERS" || return 2
  fi
  date -Is > "$d/.norm_done"
}

build_descriptors() {  # DS : umbilicus.json + spiral-scroll*.json that are missing, from the published m7 prediction (label-free)
  local ds=$1 d s z0 z1 st stage raw qa f
  d=$(ds_dir "$ds"); s=${SC[$ds]}; read -r z0 z1 <<< "${BAND[$ds]}"
  [ -e "$d/.desc_done" ] && return 0
  # The repo's automatic umbilicus for this scroll (umbilici/<scroll>/umbilicus.json: VC3D estimator with un-crush, validated
  # against hand-made axes and in downstream fits, tools/qa/umbilicus) is used when its control points span the band;
  # UMB_SOURCE=auto keeps the m7-skeleton estimate below instead.
  local rep=${UMB_REPO_DIR:-$CLOUD/../../umbilici}/$s/umbilicus.json
  if [ "${UMB_SOURCE:-repo}" = repo ] && [ -s "$rep" ] && [ ! -e "$d/umbilicus.json" ] && "$PY" -c "
import json, sys; z = [p['z'] for p in json.load(open(sys.argv[1]))['control_points']]
sys.exit(0 if min(z) <= int(sys.argv[2]) and max(z) >= int(sys.argv[3]) else 1)" "$rep" "$z0" "$z1"; then
    cp -f "$rep" "$d/umbilicus.json"; echo "umbilicus: the repo's axis $rep"
  fi
  if [ -s "$d/umbilicus.json" ] && [ -s "$d/spiral-scroll.json" ]; then date -Is > "$d/.desc_done"; return 0; fi   # made by hand / uploaded
  stage=$d/.desc_build; raw=$QW/out/umb_raw_$ds.json; qa=$RESULTS/$ds/descriptors/umbilicus_qa.png
  st=$(( (z1 - z0) / 14 )); [ $st -ge 1 ] || st=1
  rm -rf "$stage"; mkdir -p "$stage" "$QW/out" "$QW/tmp" "$(dirname "$qa")"
  if [ "$DRY" = 1 ]; then
    $MOCK inputs desc "$stage" "$s" "$z0" "$z1" "$ds" "$qa" || return 2
  else
    nice -n 19 "$PY" -u "$LQ/umb_auto.py" "$s" "$raw" "$z0" "$z1" "$st" --normals m7st --seeds 1 2 3 --uncrush "${UMB_UNCRUSH:-0.5}" || return 2
    nice -n 19 "$PY" -u "$LQ/umb_smooth.py" "$s" "$raw" "$stage" "$z0" "$z1" --qa "$qa" || return 2
    cp -f "$raw" "$(dirname "$qa")/umbilicus_raw.json"
  fi
  for f in umbilicus.json spiral-scroll.json spiral-scroll.CW.json spiral-scroll.ACW.json; do
    if [ -e "$stage/$f" ] && [ ! -e "$d/$f" ]; then mv "$stage/$f" "$d/$f"; fi    # an existing descriptor is never overwritten
  done
  rm -rf "$stage"
  [ -s "$d/umbilicus.json" ] && [ -s "$d/spiral-scroll.json" ] || return 2
  date -Is > "$d/.desc_done"
}

build_umbilicus() {  # PHerc0343x only: our own band umbilicus (umb_auto lasagna route + Huber line)
  local ds=$1 d z0 z1; d=$(ds_dir "$ds"); read -r z0 z1 <<< "${BAND[$ds]}"
  [ -e "$d/.umb_done" ] && return 0
  if [ "$DRY" = 1 ]; then $MOCK inputs umb "$d" PHerc0343 "$z0" "$z1" && touch "$d/.umb_done"; return $?; fi
  mkdir -p "$QW/out" "$QW/tmp"
  if nice -n 19 "$PY" -u "$LQ/umb_auto.py" PHerc0343 "$QW/out/umb_band_$ds.json" "$z0" "$z1" $(( (z1 - z0) / 14 )) \
        --normals lasagna --las "$d/lasagna_inputs" --seeds 1 2 3 && \
     nice -n 19 "$PY" -u "$LQ/umb_smooth.py" PHerc0343 "$QW/out/umb_band_$ds.json" "$d" "$z0" "$z1" \
        --qa "$QW/out/umbilicus_qa_$ds.png"; then
    echo "umbilicus: own band estimate written"
  else
    echo "UMBILICUS FALLBACK: umb_auto/umb_smooth failed; using the 8100-9900 PHerc0343 line (the fit extrapolates it)"
    cp -f "$DATA/spiral/PHerc0343/umbilicus.json" "$d/umbilicus.json"
    cp -f "$DATA/spiral/PHerc0343trk/spiral-scroll.json" "$d/spiral-scroll.json"
    echo fallback > "$d/.umb_fallback"
  fi
  date -Is > "$d/.umb_done"
}

spec_tracks() {  # DS : point every spiral-scroll*.json at the band DBM (idempotent)
  local ds=$1 d s z0 z1; d=$(ds_dir "$ds"); s=${SC[$ds]}; read -r z0 z1 <<< "${BAND[$ds]}"
  "$PY" - "$d" "$s" "$z0" "$z1" <<'PYEOF'
import json, sys, glob, os
d, s, z0, z1 = sys.argv[1:5]
for f in glob.glob(os.path.join(d, 'spiral-scroll*.json')):
    j = json.load(open(f)); j['paths']['tracks_dbm'] = f'tracks/{s}_m7_band{z0}_{z1}.dbm'; j['paths'].pop('gradient_magnitude', None)
    json.dump(j, open(f, 'w'), indent=1)
print('spec updated', d)
PYEOF
}

build_tracks() {  # DS : stream the band's published tracks, official crossing cache, point the spec at the DBM
  local ds=$1 d s z0 z1 db; d=$(ds_dir "$ds"); s=${SC[$ds]}; read -r z0 z1 <<< "${BAND[$ds]}"
  [ -e "$d/.trk_done" ] && return 0
  db=$d/tracks/${s}_m7_band${z0}_${z1}.dbm
  if [ "$DRY" = 1 ]; then $MOCK inputs trk "$d" "$s" "$z0" "$z1" && touch "$d/.trk_done"; return $?; fi
  mkdir -p "$d/tracks" "$QW/tmp"
  [ -e "$db" ] && mv "$db" "$db.partial_$(date +%s)"          # never delete; a partial stream is moved aside
  nice -n 19 "$PY" -u "$LQ/tracks_subset.py" "$s" "$z0" "$z1" "$d" --workers "${TRACK_WORKERS:-$(( $(nproc) > 16 ? 16 : 8 ))}" --gap-kib 256 || return 2
  mkdir -p "$QW/tmp/xing_$ds"
  (cd "$SF" && nice -n 19 "$PYX" build_track_crossings.py "$db" --z-min "$z0" --z-max "$z1" \
      --temp-dir "$QW/tmp/xing_$ds") || return 2
  rm -rf "$QW/tmp/xing_$ds"
  spec_tracks "$ds"
  (cd "$SF" && "$PYX" "$LQ/trk_loadcheck.py" "$db" "$z0" "$z1") || return 2
  date -Is > "$d/.trk_done"
}

ensure_one() {
  local ds=$1 d; d=$(ds_dir "$ds")
  [ -n "${SC[$ds]:-}" ] || { status "unknown dataset $ds"; return 2; }
  mkdir -p "$d" "$QW/tmp" "$QW/out" "$QW/logs"
  exec 7> "$d/.build.lock"; flock 7
  local waited=0
  while [ -e "$d/.upload_in_progress" ]; do
    [ $waited -ge $((UPLOAD_WAIT_H * 3600)) ] && { status "$ds: upload still in progress after ${UPLOAD_WAIT_H} h; giving up"; return 3; }
    [ $((waited % 1800)) -eq 0 ] && status "$ds: waiting for the upload to finish"
    sleep 60; waited=$((waited + 60))
  done
  if ready "$ds"; then status "$ds: inputs ready"; flock -u 7; return 0; fi
  status "$ds: building missing inputs on the pod (desc=$([ -e $d/.desc_done ] || [ -e $d/.umb_done ] && echo ok || echo build) norm=$([ -e $d/.norm_done ] || [ -e $d/.upload_done ] && echo ok || echo build) trk=$([ -e $d/.trk_done ] && echo ok || echo build))"
  local t0; t0=$(date +%s)
  if [ "$ds" != PHerc0343x ]; then
    build_descriptors "$ds" || { status "$ds: descriptors FAILED (umb_auto / umb_smooth; see $STATE/logs)"; return 2; }
    [ -e "$d/.desc_done" ] && status "$ds: descriptors ready (umbilicus.json, spiral-scroll*.json; QA images in $RESULTS/$ds/descriptors)"
  fi
  if [ ! -e "$d/.upload_done" ]; then build_normals "$ds" || { status "$ds: normals FAILED"; return 2; }; fi
  if [ "$ds" = PHerc0343x ]; then build_umbilicus "$ds" || { status "$ds: umbilicus FAILED"; return 2; }; fi
  build_tracks "$ds" || { status "$ds: tracks FAILED"; return 2; }
  spec_tracks "$ds" > /dev/null
  if ready "$ds"; then status "$ds: inputs built in $(( $(date +%s) - t0 )) s"; flock -u 7; return 0; fi
  status "$ds: inputs still not ready after the build (check spiral-scroll.json / markers)"; return 2
}

case "${1:-status}" in
  status)
    for ds in "${!SC[@]}"; do d=$(ds_dir "$ds")
      printf '%-11s ready=%s norm=%s trk=%s umb=%s upload=%s\n' "$ds" "$(ready "$ds" && echo YES || echo no)" \
        "$([ -e $d/.norm_done ] && echo built || ([ -e $d/.upload_done ] && echo uploaded) || echo -)" \
        "$([ -e $d/.trk_done ] && echo ok || echo -)" "$([ -e $d/.umb_done ] || [ -e $d/.desc_done ] && echo ok || echo -)" \
        "$([ -e $d/.upload_in_progress ] && echo IN-PROGRESS || ([ -e $d/.upload_done ] && echo done) || echo -)"
    done | sort ;;
  ensure)
    shift; rc=0; for ds in "$@"; do ensure_one "$ds" || rc=$?; done; exit $rc ;;
  bg)
    shift; setsid nohup nice -n 19 bash "$0" ensure "$@" >> "$STATE/logs/inputs_bg.log" 2>&1 < /dev/null 7>&- 8>&- 9>&- &
    status "background input builds started for: $* (pid $!, log $STATE/logs/inputs_bg.log)" ;;
  *) echo "usage: inputs.sh status | ensure DS... | bg DS..."; exit 2 ;;
esac
