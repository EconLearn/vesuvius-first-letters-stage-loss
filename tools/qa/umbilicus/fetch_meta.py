"""Fetch Drobkov's manual umbilici (MIT) and, per scroll, the m7 L0 th0.2 surface-prediction zarr + its level-2 array shape/chunks."""
import json, sys, urllib.request, s3fs
from pathlib import Path
OUT = Path(sys.argv[1]); fs = s3fs.S3FileSystem(anon=True); B = 'vesuvius-challenge-open-data'
rows = {}
for s in ['PHerc0191', 'PHerc0257', 'PHerc0268', 'PHerc0358', 'PHerc0800', 'PHerc0813', 'PHerc1203', 'PHerc1218', 'PHerc1447', 'PHerc1545']:
    f = OUT / 'ref' / f'{s}_umbilicus.json'
    if not f.exists():
        try:
            urllib.request.urlretrieve(f'https://raw.githubusercontent.com/AlexeyDrobkovStrikesBack/herculaneum-umbilici/main/{s}_umbilicus.json', f)
        except Exception as e:
            print(s, 'no umbilicus file', e); continue
    j = json.load(open(f)); md = j.get('metadata', {}); vid = md.get('volume') or md.get('source_volume', '').split('/')[-1]
    stamp = vid.split('-')[0]; vox = md.get('voxelsize_um')
    try:
        names = [p.rstrip('/').split('/')[-1] for p in fs.ls(f'{B}/{s}/representations/predictions/surfaces/')]
    except Exception as e:
        print(s, 'no surfaces listing', e); continue
    cand = sorted(n for n in names if n.endswith('-surface-m7-L0-th0.2.zarr') and n.startswith(stamp))
    if not cand:
        print(s, 'no m7 L0 for', stamp, names[:4]); continue
    m7 = f'{B}/{s}/representations/predictions/surfaces/{cand[-1]}'
    za = json.loads(fs.cat_file(f'{m7}/2/.zarray'))
    zs = [p['z'] for p in j['control_points']]
    rows[s] = dict(vol=vid, vox_um=vox, m7=m7, l2_shape=za['shape'], l2_chunks=za['chunks'], sep=za.get('dimension_separator', '.'),
                   n_points=len(zs), z_range=[min(zs), max(zs)])
    print(s, json.dumps(rows[s])[:300])
(OUT / 'scrolls.json').write_text(json.dumps(rows, indent=1)); print(len(rows), 'scrolls usable')
