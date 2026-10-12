#!/bin/bash
# GPU wrapper. usage: gpu_locked.sh LABEL CMD [ARGS...]
#  1. waits for any live lock held by someone else, then creates $GPU_LOCK atomically
#     ("<my pid> <label> <time>"), so the fit supervisor (quiet/quiet_super.py) launches/resumes no fit meanwhile;
#  2. waits 30 s (any launch already past its last lock check shows up), then until no fit_spiral.py runs and no
#     supervisor holds a started job that could resume (pending_resume.py), then
#     until GPU memory used <= 1500 MiB (nothing else on the card);
#  3. runs CMD at normal priority; 4. removes the lock in a trap (only if it is still ours).
# Never starts GPU work while a fit is running. Kills nothing.
# env: VC_ROOT (default ~/vesuvius), VC_WORK, QUIET_WORK (default $VC_WORK/quiet), GPU_LOCK (default $QUIET_WORK/GPU_LOCK),
#      GPU_LOCK_LOG (default $QUIET_WORK/gpu_lock.log), POD_PY (python, default $VC_ROOT/.venv/bin/python)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VC_ROOT=${VC_ROOT:-$HOME/vesuvius}                       # see tools/vcpaths.py
VC_WORK=${VC_WORK:-$VC_ROOT/work}
QUIET_WORK=${QUIET_WORK:-$VC_WORK/quiet}
export VC_ROOT VC_WORK QUIET_WORK                        # pending_resume.py reads the same locations
PYV=${POD_PY:-$VC_ROOT/.venv/bin/python}
LOCK=${GPU_LOCK:-$QUIET_WORK/GPU_LOCK}
LOG=${GPU_LOCK_LOG:-$QUIET_WORK/gpu_lock.log}
mkdir -p "$(dirname "$LOCK")" "$(dirname "$LOG")"
LABEL=$1; shift
say() { echo "$(date -Is) [$LABEL pid $$] $*" | tee -a $LOG; }
mine() { [ -e $LOCK ] && [ "$(awk '{print $1; exit}' $LOCK 2>/dev/null)" = "$$" ]; }
cleanup() { if mine; then rm -f $LOCK; say "GPU_LOCK removed"; fi; }
trap cleanup EXIT
trap 'say "signal; exiting"; exit 130' INT TERM HUP
while true; do
  if ( set -C; echo "$$ $LABEL $(date -Is)" > $LOCK ) 2>/dev/null; then break; fi
  p=$(awk '{print $1; exit}' $LOCK 2>/dev/null)
  if [[ "$p" =~ ^[0-9]+$ ]] && [ ! -d /proc/$p ]; then say "stale lock of dead pid $p removed"; rm -f $LOCK; continue; fi
  say "lock held: $(cat $LOCK 2>/dev/null); waiting"; sleep 60
done
say "GPU_LOCK created"
sleep 30
n=0
# no fit may run, and no supervisor may hold a started (partial) job that could resume: resumes ignore the lock
while pgrep -f 'fit_spiral.py' > /dev/null || [ -n "$("$PYV" "$HERE/pending_resume.py" 2>/dev/null)" ]; do
  [ $((n % 10)) -eq 0 ] && say "waiting: fits [$(pgrep -f fit_spiral.py | tr '\n' ' ')] pending resumes [$("$PYV" "$HERE/pending_resume.py" 2>/dev/null)]"; n=$((n+1)); sleep 60
done
n=0
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 1500 ]; do
  [ $((n % 20)) -eq 0 ] && say "waiting for GPU memory to drain: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; n=$((n+1)); sleep 30
done
say "GPU free; running: $*"
t0=$(date +%s)
"$@"
rc=$?
say "done rc=$rc in $(( $(date +%s) - t0 )) s"
exit $rc
