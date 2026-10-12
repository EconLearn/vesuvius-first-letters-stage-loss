#!/usr/bin/env python
"""tools/ensemble/render66.py unchanged, with the screened scrolls of screen_scrolls.tsv added to ens.common.VOLUMES
in-process (see volumes.py) and the CT .zarray fetched first. Arguments are passed through unchanged.
usage: r66.py [render66 args...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import volumes  # noqa: E402

C = volumes.register()
import render66  # noqa: E402  (tools/ensemble is on sys.path after register())

if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--scroll' in argv:
        volumes.ensure_zarray(argv[argv.index('--scroll') + 1])
    sys.exit(render66.main(argv))
