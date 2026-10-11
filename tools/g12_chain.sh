#!/bin/bash
# Downstream chain for one fit snapshot: official flatten -> render (flat + raw) -> ink_9um infer -> score.
# usage: g12_chain.sh SNAPTAG CKPT          (env RAW=0 skips the unflattened arm)
# Env: VC_ROOT (default ~/vesuvius), PY (python with villa `vesuvius` installed), SPIRAL_PY (python for spiral-fitting).
VC_ROOT=${VC_ROOT:-$HOME/vesuvius}; export VC_ROOT
VC_WORK=${VC_WORK:-$VC_ROOT/work}; VC_OUT=${VC_OUT:-$VC_ROOT/out}
PY=${PY:-python}; SPIRAL_PY=${SPIRAL_PY:-python}
TOOLS=$(cd "$(dirname "$0")" && pwd)
TAG=$1; CK=$2
W=$VC_WORK/g1g2; mkdir -p "$W/flat"; cd "$W"
echo "== chain $TAG start $(date -Is)"
t=$(date +%s)
if [ ! -f flat/$TAG.tifxyz/x.tif ]; then
  # on a shared 8 GB card: wait for 2.5 GB free (and util <= 90%) before the flatten service allocates
  "$PY" -c "import sys; sys.path.insert(0, '$TOOLS'); from vlib.guards import gpu_check; gpu_check(2500, wait_s=6*3600, max_util=90)"
  "$SPIRAL_PY" "$TOOLS/g12_flatten.py" "$CK" "$W/flat/$TAG.tifxyz" 2>&1 | grep -v -E 'GET /jobs|\] loading$|starting$'
fi
echo "TIME flatten $(( $(date +%s)-t ))s"; t=$(date +%s)
"$PY" "$TOOLS/g12_g2.py" auto "$W/flat/$TAG.tifxyz" ${TAG}flat --procs 5 || exit 1
TAGS="${TAG}flat"
if [ "${RAW:-1}" = "1" ]; then
  "$PY" "$TOOLS/g12_g2.py" auto "$W/flat/$TAG.source.tifxyz" ${TAG}raw --procs 5 || exit 1
  TAGS="${TAG}flat ${TAG}raw"
fi
echo "TIME render $(( $(date +%s)-t ))s"; t=$(date +%s)
N=$("$PY" -c "import json,sys;print(' '.join(c['name'] for t in sys.argv[1:] for c in json.load(open('$W/arms/auto_'+t+'.json'))['comps']))" $TAGS)
"$PY" "$TOOLS/g12_g2.py" infer $N || exit 1
echo "TIME infer $(( $(date +%s)-t ))s"; t=$(date +%s)
"$PY" "$TOOLS/g12_g2.py" score $TAGS 2>&1 | grep -v -i warn
cp "$VC_OUT/g1g2/g2_scores.json" "$VC_OUT/g1g2/g2_scores_$TAG.json"
echo "TIME score $(( $(date +%s)-t ))s"
echo "== chain $TAG done $(date -Is)"
