"""Fetch the published m7 xy normal grids of every reference scroll at a fixed physical stride (about 2.4 mm) over the
reference umbilicus' z range (2 %..98 %). Writes DATA/grids/<scroll>/<z>.grid and DATA/scrolls.json.
usage: fetch_grids.py DATA"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import s3
D = Path(sys.argv[1]); L = json.loads((D / 'listing.json').read_text())
# Grid level factor (reference z / grid z) and grid z stride. 0332 and 1451 have level-2 grids of a 2.4 um scan;
# Paris4's grids are level 0 of its 2.4 um scan. PHerc0139 is left out: its umbilicus is on another scan than its grids.
CFG = {'PHerc0125': 1, 'PHerc0211': 1, 'PHerc0826': 1, 'PHerc0332': 4, 'PHerc1451': 4, 'PHercParis4': 1,
       'PHerc0191': 1, 'PHerc0257': 1, 'PHerc0268': 1, 'PHerc0358': 1, 'PHerc0800': 1, 'PHerc0813': 1, 'PHerc1203': 1,
       'PHerc1218': 1, 'PHerc1447': 1, 'PHerc1545': 1}
STRIDE = {'PHercParis4': 1024}
out = {}; jobs = []
for s, f in CFG.items():
    refk = 'drobkov' if 'drobkov' in L[s]['refs'] else 'challenge0'
    u = json.loads(Path(L[s]['refs'][refk]).read_text()); md = u['metadata']
    zs = sorted(p['z'] for p in u['control_points']); z0, z1 = zs[0], zs[-1]
    st = STRIDE.get(s, 256); off = st // 2; lo = (z0 + 0.02 * (z1 - z0)) / f; hi = (z0 + 0.98 * (z1 - z0)) / f
    gz = list(range(int(lo // st + 1) * st + off, int(hi) + 1, st))
    ng = L[s]['normal_grids'][0]
    out[s] = dict(ref=L[s]['refs'][refk], ref_kind=refk, level_factor=f, vox_um=md['voxelsize_um'], stride=st, z=gz, ng=ng)
    (D / 'grids' / s).mkdir(parents=True, exist_ok=True)
    jobs += [(ng, s, z) for z in gz]
def one(j):
    ng, s, z = j; p = D / 'grids' / s / f'{z:06d}.grid'
    if not p.exists(): p.write_bytes(s3.cat(f'{ng}/xy/{z:06d}.grid'))
    return p.stat().st_size
print(len(jobs), 'grids', flush=True)
with ThreadPoolExecutor(8) as ex: sizes = list(ex.map(one, jobs))
print('MB', round(sum(sizes) / 1e6, 1))
(D / 'scrolls.json').write_text(json.dumps(out, indent=1))
for s, r in out.items(): print(s, r['ref_kind'], len(r['z']), 'slices, stride', r['stride'])
