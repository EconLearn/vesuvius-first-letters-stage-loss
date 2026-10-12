#!/usr/bin/env python
"""m7path.py SCROLL : print the s3:// path of the scroll's published m7 surface prediction (binary sheet mask, L0,
threshold 0.2), for snap_meshes_nieuwlaar.py. Lists <scroll>/representations/predictions/surfaces/ on the public bucket
(anonymous) and picks the m7 L0 th0.2 zarr of the scroll's configured CT volume; exits 1 if there is none."""
import sys
from pathlib import Path

import s3fs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volumes  # noqa: E402

BUCKET = 'vesuvius-challenge-open-data'


def m7_path(scroll):
    reg = {**getattr(volumes, 'SCREEN', {}), **volumes.KNOWN}
    vid = reg.get(scroll, (None,))[0]
    stamp = vid.split('-')[0] if vid else None
    fs = s3fs.S3FileSystem(anon=True)
    names = [p.rstrip('/').split('/')[-1] for p in fs.ls(f'{BUCKET}/{scroll}/representations/predictions/surfaces/')]
    cand = sorted(n for n in names if n.endswith('-surface-m7-L0-th0.2.zarr') and (stamp is None or n.startswith(stamp)))
    return f's3://{BUCKET}/{scroll}/representations/predictions/surfaces/{cand[-1]}' if cand else None


if __name__ == '__main__':
    p = m7_path(sys.argv[1])
    if not p:
        sys.exit(f'{sys.argv[1]}: no m7 L0 th0.2 surface prediction found')
    print(p)
