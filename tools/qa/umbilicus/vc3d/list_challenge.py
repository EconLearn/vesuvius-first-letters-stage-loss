import s3fs, json, sys
fs = s3fs.S3FileSystem(anon=True); B = 'vesuvius-challenge-open-data'
tops = [p.split('/')[-1] for p in fs.ls(B) if p.split('/')[-1].startswith('PHerc')]
out = {}
for s in tops:
    try: u = fs.ls(f'{B}/{s}/representations/umbilicus')
    except Exception: continue
    js = [p for p in u if p.endswith('.json')]
    try: ng = [p for p in fs.ls(f'{B}/{s}/representations/predictions/surfaces') if p.endswith('.normal-grids')]
    except Exception: ng = []
    out[s] = {'umb': js, 'ng': ng}; print(s, [p.split('/')[-1] for p in js], [p.split('/')[-1] for p in ng], flush=True)
json.dump(out, open(sys.argv[1] + '/listing.json', 'w'), indent=1)
