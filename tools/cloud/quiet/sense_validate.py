#!/usr/bin/env python
"""Prepare the known-answer check of the winding-sense rule (SENSE_VALIDATE=1 in jobs/job3_quiet.sh).

The check fits a band whose catalogue sense is known (default: the PHerc0343 stage-2 band, z_top_to_bottom = false) once per
sense, exactly like an unknown-sense band, and requires quiet/pick_sense.py to choose the catalogue sense.

  sense_validate.py variants --dataset DATASET_DIR --out DIR
      Reads DATASET_DIR/spiral-scroll.json, which must carry the catalogue properties (z_direction_is_top_to_bottom not
      null). Writes DIR/spiral-scroll.CW.json and DIR/spiral-scroll.ACW.json: the same spec with the catalogue properties
      set to null (the fit refuses a spiral_outward_sense that contradicts them) and spiral_outward_sense = CW / ACW; paths
      stay relative to the dataset root, so the fits are started with --dataset DATASET_DIR --scroll-spec DIR/<variant>.
      Also writes DIR/expected_sense: the catalogue sense, ACW if z_direction_is_top_to_bottom != left_handed_coordinates,
      else CW (the rule of spiral-fitting's surface_orientation.spiral_outward_sense_for).
"""
import argparse
import json
import sys
from pathlib import Path


def catalogue_sense(z_top_to_bottom, left_handed=False):
    return 'ACW' if bool(z_top_to_bottom) != bool(left_handed) else 'CW'


def variants(dataset, out):
    src = Path(dataset) / 'spiral-scroll.json'
    spec = json.loads(src.read_text())
    z = spec.get('z_direction_is_top_to_bottom')
    if z is None:
        sys.exit(f'{src}: no z_direction_is_top_to_bottom: not a band of known sense, cannot be used for the known-answer check')
    expected = catalogue_sense(z, spec.get('left_handed_coordinates') or False)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec.pop('spiral_outward_sense', None)
    spec.pop('_sense_note', None)
    spec['z_direction_is_top_to_bottom'] = None
    spec['left_handed_coordinates'] = None
    for sn in ('CW', 'ACW'):
        (out / f'spiral-scroll.{sn}.json').write_text(json.dumps(dict(spec, spiral_outward_sense=sn), indent=1))
    (out / 'expected_sense').write_text(expected + '\n')
    print(f'validation variants for {Path(dataset).name} written to {out}; catalogue z_top_to_bottom={z} -> expected sense {expected}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('variants'); v.add_argument('--dataset', required=True); v.add_argument('--out', required=True)
    a = ap.parse_args()
    return variants(a.dataset, a.out)


if __name__ == '__main__':
    sys.exit(main())
