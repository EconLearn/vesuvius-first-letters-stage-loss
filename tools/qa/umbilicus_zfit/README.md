# `vc_gen_umbilicus --zfit`: benchmark

`vc_gen_umbilicus` estimates each slice's umbilicus on its own, and the estimate is randomised. `--zfit H` (villa PR,
link in the PR body) fits one robust local-linear curve (LOWESS) along z through every estimate. This folder is the
evaluation behind that PR: data fetch, raw estimates from villa's own C++ estimator, the fit, the scoring, and the
pre-registration (`PREREG_zfit.txt`, written before any fitted error was computed; results are appended to it in order).

Result (repeats 1, uncrush 1.5, 13 held-out scrolls, median of per-scroll median distance to the reference axis):

| Slice set | Today | `--zfit` 4.7 mm | Scroll-level p | Worst scroll |
|---|---|---|---|---|
| First (z = 256 k, seeds 1..9, 788 slices) | 1.29 mm | 1.03 mm | 0.005 | +0.24 mm |
| Replication (z = 128 + 256 k, seeds 11..19, 781 slices) | 1.36 mm | 1.07 mm | 0.004 | +0.04 mm |

Without uncrush the median gain held (2.11 → 1.63 and 1.95 → 1.62 mm), but the replication failed the pre-registered
per-scroll limit (PHerc0268 +0.35 mm). Logs: `results/qa/umbilicus_zfit_first.log`, `..._replication.log`. Per-slice
errors for every arm: the matching `.json` files.

Run:
```
python listing.py DATA                       # references (herculaneum-umbilici, MIT; challenge umbilici) + grid paths
python fetch_grids.py DATA                   # 962 grids, 355 MB, anonymous HTTPS (fetch_grids_rep.py: the replication set)
zsh build_all.sh VILLA_CHECKOUT              # needs inc/nlohmann/json.hpp (v3.11.3, sha256 9bea4c80...103ea6)
clang++ ... evaldrv.cpp                      # see build_all.sh; links villa's GridStore and normalgridtools
python run_raw.py DATA ./evaldrv 4           # raw estimates only, no reference comparison
python score.py DATA dev                     # chooses H on PHerc0358/1203/0191
python score.py DATA final 4.681             # held-out table, ship rule, sabotage (z-shuffled fit)
```
`zfit.py` is the reference implementation. The C++ `fit_umbilicus_along_z` matches it to 5e-10 voxels (`zfitdrv.cpp`).
`cvshim/` is a minimal OpenCV stand-in for building on a Mac without OpenCV. The known answer from the merged
`--uncrush` PR reproduces with it: PHerc1447 z 16030, 1.354 → 0.290 mm.
