"""Path configuration shared by every tool in this repository (no machine-specific paths in the code).

Set VC_ROOT to a working directory (default: ~/vesuvius). Each sub-path can be overridden on its own.

  $VC_DATA  = $VC_ROOT/data
      hf/ink-8um-pherc1447-surfaces/      HF dataset YoussefMoNader/ink-8um-pherc1447-surfaces (w058, w060, w062)
      s3/<key>                            local mirror of s3://vesuvius-challenge-open-data/<key> (filled on demand)
      spiral/PHerc1447/                   spiral-fit dataset dir (spiral-scroll.json, umbilicus.json, lasagna_inputs/)
      hf_bucket/scrollprize_datasets/     HF bucket scrollprize/datasets (only ink/unused/P343p is read)
  $VC_CKPT  = $VC_ROOT/ckpt
      ink_9um/hybrid_3d2d-seed42/step-075000.pth, ink_9um/hybrid_3d2d-seed43/step-050000.pth   (scrollprize/ink_9um)
      v8in/                               YoussefMoNader/ink-8um-v8in (the ink8um package + weights)
  $VC_WORK  = $VC_ROOT/work               intermediate files (renders, predictions, fits)
  $VC_OUT   = $VC_ROOT/out                result JSONs
  $VILLA_DIR       = $VC_ROOT/repos/villa                 ScrollPrize/villa checkout (tested at e0bbb8b4)
  $SPIRAL_FIT_DIR  = $VILLA_DIR/spiral-fitting           (surf_eval.py needs the vc_spiral native extensions built)
  $V8IN_DIR        = $VC_CKPT/v8in
  $VC_DISK_GUARD_PATH, $VC_MIN_FREE_GB   disk guard used before large fetches (default: VC_ROOT's disk, 20 GB)
"""
import os

ROOT = os.environ.get('VC_ROOT', os.path.join(os.path.expanduser('~'), 'vesuvius'))
DATA = os.environ.get('VC_DATA', os.path.join(ROOT, 'data'))
WORK = os.environ.get('VC_WORK', os.path.join(ROOT, 'work'))
OUT = os.environ.get('VC_OUT', os.path.join(ROOT, 'out'))
CKPT = os.environ.get('VC_CKPT', os.path.join(ROOT, 'ckpt'))
VILLA = os.environ.get('VILLA_DIR', os.path.join(ROOT, 'repos', 'villa'))
SPIRAL_FIT_DIR = os.environ.get('SPIRAL_FIT_DIR', os.path.join(VILLA, 'spiral-fitting'))
V8IN_DIR = os.environ.get('V8IN_DIR', os.path.join(CKPT, 'v8in'))
TOOLS = os.path.dirname(os.path.abspath(__file__))
MIN_FREE_GB = float(os.environ.get('VC_MIN_FREE_GB', '20'))


def disk_guard_path():
    """First existing of $VC_DISK_GUARD_PATH, $VC_DATA, $VC_ROOT, home."""
    for p in (os.environ.get('VC_DISK_GUARD_PATH'), DATA, ROOT, os.path.expanduser('~')):
        if p and os.path.exists(p):
            return p
    return '/'
