"""CT volume registry for the cloud jobs: the two scrolls ens/common.VOLUMES already knows plus the screened scrolls of
screen_scrolls.tsv ($SCREEN_CONFIG), with vol_id and shape taken from the public S3 listing. register() adds the missing
entries to ens.common.VOLUMES in-process (no repository file is edited); ensure_zarray() writes the tiny .zarray that
ens.common.use_volume() reads before any chunk is fetched (anonymous public S3).

Paths follow tools/vcpaths.py: CT mirror $VC_DATA/s3/<scroll>/volumes/<vol_id>/0, umbilicus
$VC_DATA/spiral/<scroll>/umbilicus.json. The ensemble package is found through $ENS_LANE (default tools/ensemble)."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]                                  # tools/: vcpaths.py
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)

ENS = Path(os.environ.get('ENS_LANE', str(TOOLS / 'ensemble')))
CONFIG = Path(os.environ.get('SCREEN_CONFIG', str(HERE.parent / 'screen_scrolls.tsv')))
KNOWN = {
    'PHerc0343': ('20250521140437-8.640um-1.2m-116keV-masked.zarr', (17998, 8595, 8595)),
    'PHerc1447': ('20250521151220-8.640um-1.2m-116keV-masked.zarr', (24297, 8343, 8343)),
}


def load_config(path=CONFIG):
    """screen_scrolls.tsv -> {scroll: (vol_id, (z, y, x))} for the rows that carry volume metadata."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#') or line.startswith('dataset\t'):
            continue
        f = line.split('\t')
        scroll, vid = f[1], f[7]
        if vid != '-' and scroll not in out:
            out[scroll] = (vid, (int(f[8]), int(f[9]), int(f[10])))
    return out


def load_voxel(path=CONFIG):
    """screen_scrolls.tsv -> {scroll: voxel size in um} (column voxel_um; 8.64 when absent or '-')."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#') or line.startswith('dataset\t'):
            continue
        f = line.split('\t')
        if len(f) > 12 and f[12] not in ('', '-'):
            out.setdefault(f[1], float(f[12]))
    return out


SCREEN = load_config()
VOXEL = load_voxel()


def register(C=None):
    if C is None:
        if str(ENS) not in sys.path:
            sys.path.insert(0, str(ENS))
        from ens import common as C
    for s, (vid, shp) in SCREEN.items():
        C.VOLUMES.setdefault(s, dict(vol_id=vid, shape=shp, um=VOXEL.get(s, 8.64),
                                     umbilicus=str(Path(_P.DATA) / 'spiral' / s / 'umbilicus.json')))
    return C


def mirror_dir(scroll):
    vid, _ = {**SCREEN, **KNOWN}[scroll]
    return Path(_P.DATA) / 's3' / scroll / 'volumes' / vid / '0'


def ensure_zarray(scroll, dry=False):
    """Write <mirror>/.zarray from public S3 if absent and check it against the registry shape."""
    vid, shp = {**SCREEN, **KNOWN}[scroll]
    m = mirror_dir(scroll)
    za = m / '.zarray'
    if not za.exists():
        if dry:
            print(f'[dry] would fetch s3://vesuvius-challenge-open-data/{scroll}/volumes/{vid}/0/.zarray -> {za}')
            return None
        import s3fs
        m.mkdir(parents=True, exist_ok=True)
        fs = s3fs.S3FileSystem(anon=True)
        za.write_bytes(fs.cat_file(f'vesuvius-challenge-open-data/{scroll}/volumes/{vid}/0/.zarray'))
    j = json.loads(za.read_text())
    ok = (tuple(j['shape']) == tuple(shp) and tuple(j['chunks']) == (128, 128, 128) and j['compressor'] is None)
    if not ok:
        raise SystemExit(f'{scroll}: .zarray {j} does not match the registry {shp} / uncompressed 128^3')
    return j


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('scrolls', nargs='+')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    for s in a.scrolls:
        print(s, ensure_zarray(s, a.dry))
