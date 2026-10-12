"""Official-loader check of a streamed band track DBM (spiral-fitting tracks.load_tracks_from_dbm + crossing cache)."""
import os, sys, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))    # tools/: vcpaths.py
import vcpaths as _P  # noqa: E402  (path configuration, see tools/vcpaths.py)
sys.path.insert(0, _P.SPIRAL_FIT_DIR)
from tracks import load_tracks_from_dbm, load_track_crossing_cache
dbm, z0, z1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = time.time()
tr, fam = load_tracks_from_dbm(dbm, z0, z1, return_families=True, show_progress=False)
print('official loader:', len(tr), f'tracks in [{z0}, {z1});', dict(collections.Counter(list(fam))), 'in', round(time.time() - t), 's')
c = load_track_crossing_cache(dbm)
print('crossing cache loaded and matches the DBM fingerprint:', c is not None)
