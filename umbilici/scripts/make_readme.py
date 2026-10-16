"""Write umbkit/README.md from summary.json (+ notes in scripts/readme_notes.md, written by hand after looking at qa.png)."""
import json
from pathlib import Path
W = Path(__file__).resolve().parents[1]
S = json.loads((W / 'summary.json').read_text())
order = ['PHerc0175A', 'PHerc0175B', 'PHerc0306B', 'PHerc0483A', 'PHerc0483B', 'PHerc0490A', 'PHerc0490B', 'PHerc0846A',
         'PHerc0846A/alt_volume_20260319102732-2.403um', 'PHerc0846B']
notes = (W / 'scripts' / 'readme_notes.md').read_text() if (W / 'scripts' / 'readme_notes.md').exists() else ''


def kit(s):
    if s['kit'] is None: return 'no kit route'
    if not s['kit_n']: return f"no overlap (kit band z {s['kit_band'][0]}-{s['kit_band'][1]})"
    return f"{s['kit']['normal']:.1f} mm (n={s['kit_n']}, max {s['kit_max']:.1f})"


rows = []
for u in order:
    if u not in S: continue
    s = S[u]
    rows.append(f"| {u} | `{s['volume']}` | {s['vox_um']} | L{s['grid_level']} (x{s['f']}) | {s['z_range'][0]}-{s['z_range'][1]} | "
                f"{s['n_used']} / {s['n_slices_total']} | {s['n_points']} ({s['cp_z_range'][0]}-{s['cp_z_range'][1]}) | "
                f"{s['unsure']['normal']} / {s['unsure_in_range']['normal']} | {s['rejected_in_range']['normal']} | {s['n_trimmed']['normal']} | "
                f"{s['median_spread_mm']['normal']:.2f} | {kit(s)} |")
table = '\n'.join(rows)
rows2 = []
for u in order:
    if u not in S: continue
    s = S[u]; g = s['method_gap_mm']
    rows2.append(f"| {u} | {s['unsure']['unchanged']} / {s['unsure']['midpoint']} / {s['unsure']['normal']} | "
                 f"{s['rejected']['unchanged']} / {s['rejected']['midpoint']} / {s['rejected']['normal']} | "
                 f"{s['median_spread_mm']['unchanged']:.2f} / {s['median_spread_mm']['midpoint']:.2f} / {s['median_spread_mm']['normal']:.2f} | "
                 f"{g['normal-unchanged']:.1f} / {g['normal-midpoint']:.1f} / {g['midpoint-unchanged']:.1f} | "
                 + (' / '.join('-' if s['kit'][m] is None else f"{s['kit'][m]:.1f}" for m in ('unchanged', 'midpoint', 'normal')) if s['kit'] and s['kit_n'] else '-') + ' |')
table2 = '\n'.join(rows2)

README = f"""# Automatic umbilici for the First Letters scrolls that have none

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
| `<SCROLL>/umbilicus.json` | primary method (normal un-crush, beta 1.5), smoothed control points `{{x, y, z, score}}` in the volume's full-resolution voxel frame, metadata in the official schema plus our keys |
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
{table}

All three methods side by side (MEASURED; order unchanged / midpoint / normal). "Method gap" is the median distance between
two methods' raw per-slice estimates.

| unit | UNSURE | rejected | median seed spread mm | method gap mm (N-U / N-M / M-U) | kit agreement mm (U / M / N) |
|---|---|---|---|---|---|
{table2}

{notes}

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
"""
(W / 'README.md').write_text(README, encoding='utf-8')
print('README.md', len(README), 'chars')
