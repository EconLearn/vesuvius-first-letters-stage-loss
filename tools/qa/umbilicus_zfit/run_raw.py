"""Raw per-slice umbilicus estimates (grid units, x y) of the C++ estimator on every fetched slice, for each uncrush
value and seeds 1..9. Saves DATA/raw.json; computes NO errors (scoring is separate, after pre-registration).
usage: run_raw.py DATA HARNESS [jobs]"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
D = Path(sys.argv[1]); H = sys.argv[2]; J = int(sys.argv[3]) if len(sys.argv) > 3 else 6
S = json.loads((D / 'scrolls.json').read_text()); BETAS = ('0', '1.5'); SEEDS = range(1, 10)
jobs = [(s, b, sd) for s in S for b in BETAS for sd in SEEDS]
def one(j):
    s, b, sd = j
    files = [str(D / 'grids' / s / f'{z:06d}.grid') for z in S[s]['z']]
    r = subprocess.run([H, b, str(sd)] + files, capture_output=True, text=True, check=True).stdout.split('\n')
    pts = [list(map(float, l.split())) for l in r if l.strip()]
    assert len(pts) == len(files), (s, b, sd, len(pts))
    return j, pts
out = {s: {b: {} for b in BETAS} for s in S}
with ThreadPoolExecutor(J) as ex:
    for n, ((s, b, sd), pts) in enumerate(ex.map(one, jobs)):
        out[s][b][str(sd)] = pts
        if n % 24 == 0: print(n, '/', len(jobs), flush=True)
(D / 'raw.json').write_text(json.dumps(out))
print('done')
