# Automatic umbilici for the First Letters scrolls that have none

**What this is.** An automatic scroll axis (umbilicus) for each of the 9 eligible First Letters scrolls that have no umbilicus
from anyone (no official file on S3, none in Drobkov's hand-made set): PHerc0175A, 0175B, 0306B, 0483A, 0483B, 0490A, 0490B,
0846A, 0846B. PHerc0846A has a second volume (2.403 um) with its own m7 normal grid, so it gets a second axis in a subfolder.
Every scroll is a top-level prefix on `s3://vesuvius-challenge-open-data/` and has a published m7 normal grid, so none was skipped.

**What it is not.** These are automatic estimates, not hand annotations. Accuracy is known only from other scrolls (below);
on these scrolls the only checks are seed agreement, slice-to-slice consistency and a comparison with a second automatic route.
Look at each `qa.png` before using a file. Nothing here has been published or uploaded.

## Files

| path | content |
|---|---|
| `<SCROLL>/umbilicus.json` | primary method (normal un-crush, beta 1.5), smoothed control points `{x, y, z, score}` in the volume's full-resolution voxel frame, metadata in the official schema plus our keys |
| `<SCROLL>/umbilicus_all_methods.json` | per slice: segment count, all 9 seed estimates of all 3 methods, their median, spread, cluster split, UNSURE flag, running median, rejection; smoothed curves of all 3 methods; kit-route agreement |
| `<SCROLL>/qa.png` | 6 slices (10-90 % of z) with the grid segments, raw and smoothed axis; x(z) and y(z) of all methods and the kit route |
| `summary.json` | the numbers in the tables below |
| `s3_inventory.json`, `xy_sizes.json`, `volume_metadata.json` | what was found on S3 (prefixes, volumes, zarr shapes, normal-grid file lists and sizes, scan metadata) |
| `work/<unit>.raw.json` | stage 1 output: segment counts and every seed's raw estimate (grid coordinates) for every slice |
| `qa_grids/` | the 6 normal-grid slices per unit drawn in `qa.png` (copies of the public S3 files) |
| `scripts/`, `harness/` | the pipeline and the estimator source (see Reproduction) |

## Per-scroll results (all MEASURED on this run, 2026-10-15; primary method)

z range = full-resolution z of the first and last slice used. Slices every 200 full-res z (every 50 grid slices for the L2 grid);
slices with < 500 segments skipped. UNSURE = seed spread > 2 mm or the 9 seeds split into two clusters (each >= 2 seeds) more
than 3 mm apart. Rejected = more than 3 mm from the 5-slice running median (rejected slices are not control points). End trim: control
points start at the first and stop at the last run of 3 consecutive slices that are kept and not UNSURE, and an end stretch
of at most 4 kept points separated from the rest by a jump > 5 mm is dropped; the slices outside are "trimmed" (see Known
limits, scan ends). "Inside range" counts only slices between the first and last control point.
Kit route = the kit's earlier umbilicus (m7 L2 skeleton + RANSAC port of the same VC3D function, 3 seeds, linear fit over a
~1000-slice band); the column is the median distance between the two routes over our kept slices inside that band. This is
consistency between two automatic routes, not truth.

| unit | volume | voxel um | normal grid | z range sampled | slices used / total | control points (z range) | UNSURE all / inside range | rejected inside range | end slices trimmed | median seed spread mm | agreement with kit route |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PHerc0175A | `20250521115057-8.640um-1.2m-116keV-masked.zarr` | 8.64 | L0 (x1) | 0-12600 | 63 / 64 | 43 (2600-11000) | 13 / 1 | 0 | 20 | 0.30 | 0.5 mm (n=5, max 0.7) |
| PHerc0175B | `20250521125822-8.640um-1.2m-116keV-masked.zarr` | 8.64 | L0 (x1) | 0-15800 | 79 / 80 | 52 (2000-12800) | 24 / 9 | 3 | 24 | 0.54 | 1.2 mm (n=5, max 1.7) |
| PHerc0306B | `20250521133212-8.640um-1.2m-116keV-masked.zarr` | 8.64 | L0 (x1) | 0-15800 | 80 / 80 | 55 (1600-12800) | 22 / 11 | 2 | 23 | 0.42 | 1.4 mm (n=4, max 2.1) |
| PHerc0483A | `20250521140913-8.640um-1.2m-116keV-masked.zarr` | 8.64 | L0 (x1) | 200-15800 | 78 / 80 | 57 (1800-13800) | 16 / 10 | 4 | 17 | 0.30 | 1.1 mm (n=5, max 1.4) |
| PHerc0483B | `20251124083638-8.640um-1.2m-116keV-masked.zarr` | 8.64 | L0 (x1) | 800-11600 | 53 / 57 | 42 (2200-11000) | 14 / 9 | 3 | 8 | 0.42 | 1.0 mm (n=3, max 2.2) |
| PHerc0490A | `20250521151210-8.640um-1.2m-116keV-masked.zarr` | 8.64 | L0 (x1) | 0-11600 | 58 / 59 | 33 (2800-10200) | 10 / 6 | 5 | 20 | 0.45 | 0.9 mm (n=5, max 1.6) |
| PHerc0490B | `20250521151215-8.640um-1.2m-116keV-masked.zarr` | 8.64 | L0 (x1) | 0-10600 | 50 / 54 | 27 (2000-9400) | 26 / 20 | 11 | 12 | 1.07 | 8.3 mm (n=2, max 11.2) |
| PHerc0846A | `20250728152254-9.362um-1.2m-113keV-masked.zarr` | 9.362 | L0 (x1) | 0-14000 | 68 / 71 | 48 (2200-13400) | 22 / 19 | 9 | 11 | 0.53 | no kit route |
| PHerc0846A/alt_volume_20260319102732-2.403um | `20260319102732-2.403um-0.2m-77keV-masked.zarr` | 2.403 | L2 (x4) | 0-15000 | 76 / 76 | 69 (1200-15000) | 9 / 8 | 1 | 6 | 0.34 | no kit route |
| PHerc0846B | `20250804142305-9.362um-1.2m-113keV-masked.zarr` | 9.362 | L0 (x1) | 400-13800 | 64 / 68 | 47 (1600-12800) | 24 / 23 | 10 | 7 | 0.72 | no kit route |

All three methods side by side (MEASURED; order unchanged / midpoint / normal). "Method gap" is the median distance between
two methods' raw per-slice estimates.

| unit | UNSURE | rejected | median seed spread mm | method gap mm (N-U / N-M / M-U) | kit agreement mm (U / M / N) |
|---|---|---|---|---|---|
| PHerc0175A | 11 / 8 / 13 | 12 / 11 / 11 | 0.45 / 0.35 / 0.30 | 0.5 / 0.4 / 0.2 | 0.3 / 0.3 / 0.5 |
| PHerc0175B | 20 / 16 / 24 | 20 / 22 / 25 | 0.46 / 0.55 / 0.54 | 0.7 / 0.5 / 0.2 | 1.2 / 1.2 / 1.2 |
| PHerc0306B | 23 / 19 / 22 | 17 / 14 / 19 | 0.54 / 0.47 / 0.42 | 0.6 / 0.6 / 0.3 | 1.6 / 1.6 / 1.4 |
| PHerc0483A | 14 / 17 / 16 | 16 / 19 / 16 | 0.52 / 0.38 / 0.30 | 1.1 / 0.5 / 0.6 | 1.0 / 1.1 / 1.1 |
| PHerc0483B | 14 / 14 / 14 | 8 / 9 / 7 | 0.52 / 0.58 / 0.42 | 0.8 / 0.7 / 0.4 | 1.3 / 1.4 / 1.0 |
| PHerc0490A | 17 / 15 / 10 | 25 / 19 / 18 | 0.65 / 0.64 / 0.45 | 0.8 / 0.8 / 0.4 | 0.9 / 0.8 / 0.9 |
| PHerc0490B | 26 / 23 / 26 | 17 / 19 / 19 | 1.03 / 0.77 / 1.07 | 3.4 / 3.7 / 1.1 | 3.6 / 4.5 / 8.3 |
| PHerc0846A | 26 / 31 / 22 | 20 / 18 / 18 | 0.56 / 0.57 / 0.53 | 1.6 / 1.1 / 0.4 | - |
| PHerc0846A/alt_volume_20260319102732-2.403um | 16 / 5 / 9 | 1 / 2 / 3 | 0.49 / 0.40 / 0.34 | 0.7 / 0.5 / 0.2 | - |
| PHerc0846B | 36 / 24 / 24 | 13 / 14 / 14 | 1.15 / 0.73 / 0.72 | 3.0 / 1.4 / 0.8 | - |

### Reading of each qa.png (judgement from the plots: GUESS unless a number is quoted)

Numbers quoted here are MEASURED on this run (primary method, 9 seeds per slice, slices every 200 full-res z).

- **PHerc0175A**: the best of the set. One clear spiral centre from z 2600 to 11000; 1 UNSURE slice inside that range, none
  rejected; agrees with the kit route to 0.53 mm (5 slices). Below z 2600 and above z 11000 the sections are fragments and the
  estimates scatter (trimmed).
- **PHerc0175B**: clean centre over most of z 2000-12800; the axis wanders smoothly along the scroll (control points span 1640 voxels = 14 mm in x and
  2260 voxels = 20 mm in y). 9 UNSURE inside the range; kit route 1.2 mm.
- **PHerc0306B**: clean middle (z 2000-12000); 11 UNSURE inside the range, mostly at z 4200-6400 where the section is
  crushed; kit route 1.4 mm.
- **PHerc0483A**: flattened sections around z 6000-9500 (UNSURE); a steep axis shift between z 9400 and 11000 (y moves about
  1480 voxels = 13 mm over 14 mm of z). The kit route shows the same slope (1.05 mm agreement), but both routes use the same
  kind of estimator, so this could be a real deformation or a switch between two apparent centres.
- **PHerc0483B**: large axis swings (y about 4000 -> 5400 -> 3300 -> 4800 voxels over z 2000-11000). Agreement with the kit
  route is 1.0 mm on the 3 kept slices of its band, but at z 8800-9000 our estimates jump to a second centre (rejected) and
  the kit line's slope differs from ours. Use with caution.
- **PHerc0490A**: z 4200-7800 has more than one apparent centre (x jumps between about 3000 and 4800 voxels; 5 rejected inside
  the range); clean from z 7800 to 10200, where the kit route agrees to 0.86 mm.
- **PHerc0490B**: the weakest. Crushed sections with no clear single centre; 20 UNSURE and 11 rejected among the 38 slices
  inside the control-point range; the kit route differs by 5.4 and 11.2 mm on the only 2 overlapping slices (and the kit's own
  seeds disagreed by up to 1900 voxels there). Treat this axis as a rough guide only.
- **PHerc0846A (9.362 um)**: flattened scroll; z 2200-7000 is consistent, z 7000-8400 and 9600-13400 are noisy (18 of its 19
  UNSURE and all 9 rejected slices inside the range), and the methods disagree there (normal vs unchanged median gap 1.6 mm). No kit route exists.
- **PHerc0846A, 2.403 um volume** (extra; L2 grid): clean centre from z 1200 to about 12000; noisy and UNSURE above 12000. The
  first 4 slices (z 0-600) sit consistently 8.7 mm away from the rest, which no real axis does over 1.4 mm of z; the end-stretch
  rule trims them.
- **PHerc0846B**: crushed in the middle; 23 UNSURE inside the range and 10 rejected; the three methods disagree more here than
  anywhere (normal vs unchanged median gap 3.0 mm). No kit route exists.

### Things that looked wrong, and what was done

1. **Scan ends.** At the first and last 1000-2500 slices of most scrolls the estimator locks onto container walls, mask edges or
   fragments, often with a small seed spread, so UNSURE does not catch it, and the running median (whose window is truncated
   at the ends) cannot reject it. Two end-trim rules were added to the brief's procedure (documented in every file's
   metadata; `UMB_TRIM=0` restores the plain procedure): control points start and end at a run of 3 consecutive kept,
   not-UNSURE slices, and an end stretch of at most 4 kept points that is separated from the rest by a jump of more than 5 mm
   is dropped. The second rule changed 3 files: PHerc0490A (last point z 10400), PHerc0846A (z 13600) and the 2.403 um volume
   (z 0-600).
2. **Smoothed vs raw.** A control point is the median of up to 5 neighbouring kept estimates, so where neighbours disagree it
   can sit a few mm from that slice's own raw estimate (for example PHerc0490A z 6000: 3.6 mm in y). The raw values are in
   `umbilicus_all_methods.json`.
3. **Estimator source drift.** During this run the harness source directory used for the binaries
   (`vc3dp/harness/extracted.inc`) was replaced by the upstream-patch version (normal un-crush under the ALPHA argument). The
   midpoint source behind `$UMB_HARNESS` then existed only as a binary. `harness/extracted_midpoint.inc` is a
   reconstruction; rebuilt with `harness/build.sh`, both binaries are byte-identical to the ones used (sha256 `d7f6ce5e...`
   and `734c235b...`), and 45 of 45 test runs (3 methods, 9 seeds, 4 slices including the L2 grid) gave identical output.


## The estimator and its measured accuracy

The estimator is VC3D's own `align_and_extract_umbilicus` (villa `volume-cartographer/core/src/normalgridtools.cpp`; RANSAC
over the normal-grid line segments plus weighted hill climbing), compiled stand-alone against small OpenCV stand-ins
(`harness/`). The un-crush option whitens the section before the search and maps the answer back:

| method | command | role |
|---|---|---|
| unchanged | `$UMB_HARNESS 0 SEED files...` | VC3D as published |
| midpoint un-crush | `$UMB_HARNESS 0.5 SEED files...` | whitens by the covariance of the segment midpoints, alpha 0.5 |
| normal un-crush | `UNCRUSH_N=1 $UMB_HARNESS_N 1.5 SEED files...` | **primary**; whitens by the orientation tensor of the normals, beta 1.5 |

Each slice is run with seeds 1-9 and the coordinatewise median is kept.

Accuracy against manual umbilici on other scrolls, median distance per slice, 9 seeds per slice. MEASURED 2026-10-15 by the
project's umbilicus evaluation (coordinator's runs of `vc3dp/harness/eval_n.py` and `umb/chal/run_chal.py`; quoted here, not
re-run for this kit), on scrolls that have manual umbilici, not on the scrolls in this kit:

| set | slices | unchanged | midpoint (alpha 0.5) | normal (beta 1.5) | test |
|---|---|---|---|---|---|
| Drobkov's manual umbilici, 7 held-out scrolls | 56 | 2.22 mm | 1.79 mm | **1.33 mm** | normal vs unchanged p = 0.0074; normal vs midpoint two-sided p = 0.014 |
| Challenge's own umbilici, 6 scrolls | 48 | 0.85 mm | 0.69 mm | **0.61 mm** | normal vs unchanged p = 0.11, **not significant** |
| Synthetic sections (round full, half, thinned) | - | - | round half sections worse: 2.42 -> 4.58 mm | essentially unchanged on round sections | - |

Earlier MEASURED numbers for the midpoint version alone (brief of this task): Drobkov set 2.00 mm unchanged vs 1.68 mm midpoint
(p = 0.0007); challenge set 0.85 vs 0.69 mm (p = 0.07). The challenge set is small and the gain there is not significant.

Seed spread on these scrolls (tables above) measures only how much the estimator's random draws disagree; it does not measure
the error against a true axis.

## Pipeline

1. Volume id, shape and voxel size from the volume zarr on S3 (`0/.zarray`, `metadata.json`); normal grid
   `representations/predictions/surfaces/*-surface-m7-*.normal-grids`. Grid level from the xy file count: equal to the
   volume's slice count -> L0 (grid pixels are full-res voxels); equal to slices/4 -> L2 (multiply by 4). Checked against the
   grid header width (L0 grids are as wide as the volume).
2. Slices at every multiple of 200 full-res z (50 grid slices at L2) between the first and last non-empty xy file. Slices with
   < 500 segments skipped. 3 methods x 9 seeds per slice; coordinatewise median; spread = median distance of the 9 estimates
   from their median (mm); cluster split by complete-linkage clustering into two groups.
3. Along z, per method: running median over 5 slices (truncated at the ends); points > 3 mm from it rejected. End trim
   (added after looking at the first QA plots; `UMB_TRIM=0` turns it off): the control points start at the first and end at
   the last run of 3 consecutive slices that are kept and not UNSURE; then an end stretch of at most 4 kept points that is
   separated from the rest by a jump of more than 5 mm between consecutive kept raw estimates is dropped. Control points = running median over 5 of the kept
   points inside that range, at their z. UNSURE slices inside the range are kept unless rejected. The untrimmed per-slice
   values of every method stay in `umbilicus_all_methods.json`.
4. Coordinates: x, y, z in full-resolution voxels of the named volume (x = column, y = row of the zarr slice), like the
   official files. Values are floats with one decimal; `score` is 100 for every point (it carries no confidence).

## Known limits

- **Automatic axis, not a hand annotation.** Accuracy on other scrolls is 0.6-1.3 mm median, with a tail: individual slices can be
  off by several mm, and the error on these scrolls is unmeasured.
- **Multi-centre and crushed sections.** Where a section has no single spiral centre (crushed, folded or split scrolls), the
  estimator's optimum is not unique: seeds split, neighbouring slices jump, and the running median picks one of the candidates.
  The UNSURE and rejected counts flag this; for scrolls with many of them the axis should be treated as a rough guide only.
- **Scan ends.** The first and last slices often hold only fragments plus the container wall or mask edges, which the estimator
  locks onto, often consistently (low seed spread, wrong position). The 5-slice running median cannot outvote them at the
  ends, where its window is truncated, so the end trim drops them; it can also drop a few good end slices. The axis is not
  defined by these files outside the control-point z range.
- **Midpoint un-crush hurts round-but-partial sections** (synthetic: 2.42 -> 4.58 mm on round half sections); that is one reason
  the normal un-crush is primary. The midpoint estimates are kept in `umbilicus_all_methods.json` for comparison.
- The challenge-set gain of the normal method over unchanged VC3D is not significant (p = 0.11); the Drobkov-set gain is.
- The kit-route comparison covers only a ~1000-slice band per scroll (a handful of slices) and the kit route is itself a noisy
  3-seed RANSAC line; it is a consistency check, not ground truth.

## Reproduction

Everything runs on CPU in WSL Ubuntu with Python 3 + numpy, scipy, s3fs, matplotlib (any venv).
Network: anonymous reads of public S3 only (~20-50 MB of `.grid` slices per scroll, held in `/dev/shm` and deleted after use).

Build the estimator (writes to `/tmp/umbkit_build`). `harness/main.cpp` + `extracted_midpoint.inc` gives `vc3d_harness`
(ALPHA 0 unchanged, 0.5 midpoint); `harness/main_n.cpp` + `extracted_n.inc` gives `vc3d_harness_n` (normal un-crush with
`UNCRUSH_N=1`). Rebuilt with g++ -O2 -std=c++17 here, both are byte-identical to the binaries used for the run (sha256
`d7f6ce5e...` and `734c235b...`, recorded in every file's metadata; MEASURED, plus 45 of 45 identical test outputs):

```
bash harness/build.sh /tmp/umbkit_build
```

Inventory S3 (prefixes, volumes, normal grids, xy file sizes, scan metadata):

```
FSSPEC_S3_ANON=true python scripts/s3list.py
```

```
FSSPEC_S3_ANON=true python scripts/sizes.py
```

Stage 1, per scroll: download the slices, run 3 methods x 9 seeds (at most 4 estimator processes at once), write
`work/<unit>.raw.json` and keep the 6 QA grids in `qa_grids/`:

```
FSSPEC_S3_ANON=true python scripts/umbkit.py run
```

Stage 2: flags, smoothing, `umbilicus.json`, `umbilicus_all_methods.json`, `qa.png`, `summary.json`; then this README:

```
python scripts/umbkit.py build
```

```
python scripts/make_readme.py
```

To write `umbilicus.json` from another method without recomputing, set `UMB_PRIMARY=midpoint` (or `unchanged`) for stage 2.
Paths are set at the top of `scripts/umbkit.py` (`W`, `SHM`, `H0`, `HN`, `KIT`) and in `scripts/s3list.py` / `sizes.py` (`OUT`, `W`); the kit-route comparison reads
`$VC_SPIRAL_DATA/<scroll>/umbilicus.json` and is skipped where that file does not exist.
