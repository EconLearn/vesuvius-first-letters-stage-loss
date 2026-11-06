"""Reference umbilici (Drobkov's hand-made set, MIT; the challenge's own published umbilici) and, per scroll, the published
m7 normal grids of the same volume. Writes DATA/listing.json. usage: listing.py DATA"""
import json, sys, urllib.request
from pathlib import Path
import s3
D = Path(sys.argv[1]); (D / 'ref').mkdir(parents=True, exist_ok=True)
DROB = ['PHerc0191', 'PHerc0257', 'PHerc0268', 'PHerc0358', 'PHerc0800', 'PHerc0813', 'PHerc1203', 'PHerc1218', 'PHerc1447', 'PHerc1545']
tops = [p.rstrip('/') for p in s3.ls('')[0] if p.startswith('PHerc')]
out = {}
def grids_for(s):
    try: pre, _ = s3.ls(f'{s}/representations/predictions/surfaces/')
    except Exception: return []
    return [p.rstrip('/') for p in pre if p.rstrip('/').endswith('.normal-grids')]
for s in tops:
    ng = grids_for(s)
    refs = {}
    if s in DROB:
        f = D / 'ref' / f'{s}_drobkov.json'
        if not f.exists():
            f.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/AlexeyDrobkovStrikesBack/herculaneum-umbilici/main/{s}_umbilicus.json').read())
        refs['drobkov'] = str(f)
    try: _, keys = s3.ls(f'{s}/representations/umbilicus/')
    except Exception: keys = []
    js = [k for k, _ in keys if k.endswith('.json')]
    for i, k in enumerate(js):
        f = D / 'ref' / f'{s}_challenge{i}.json'
        if not f.exists(): f.write_bytes(s3.cat(k))
        refs[f'challenge{i}'] = str(f); refs[f'challenge{i}_key'] = k
    if not refs: continue
    out[s] = {'refs': refs, 'normal_grids': ng}
    print(s, list(refs), [n.split('/')[-1] for n in ng], flush=True)
(D / 'listing.json').write_text(json.dumps(out, indent=1))
