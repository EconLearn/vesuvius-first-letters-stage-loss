# Umbilicus un-crush benchmark

The scroll-axis (umbilicus) estimator VC3D uses (`align_and_extract_umbilicus`; ported in `tools/cloud/quiet/umb_auto.py`)
assumes round windings. Crushed scrolls have elliptical windings whose sheet normals do not point at the axis. The fix
whitens each slice's sheet pixels by PCA with strength alpha, estimates the centre in that "un-crushed" frame, and maps it
back (`umb_auto.py --uncrush 0.5`; the cloud kit passes it by default, `UMB_UNCRUSH=0` restores the original).

Result on 7 held-out scrolls (56 slices), against hand-made umbilici: median error 1.77 mm -> 1.26 mm, within 1 mm 25 % -> 38 %,
29 slices better / 15 worse (one-sided Wilcoxon p = 0.021). The same stretch along random axes gives no gain (p = 0.88).
Numbers: `results/qa/umbilicus_uncrush.json`.

Run (anonymous public S3; ~80 level-2 slices of the published m7 surface predictions are cached locally):
```
python fetch_meta.py OUT          # manual umbilici (MIT, herculaneum-umbilici) + m7 paths -> OUT/scrolls.json
python umb_bench.py OUT           # original vs un-crush (alpha 0, 0.5, 1); dev scrolls choose alpha
python umb_ctrl.py OUT            # exact original estimator, alpha-0 baseline, un-crush, random-axis sabotage
```
Limits: some heavily crushed bands still fail with every method (~5 mm).

## On VC3D's own input and code (`umb_vc3d.py`, `vc3d/`)
`umb_vc3d.py` repeats the test on VC3D's input, the published normal grids (GridStore v3, read directly), with a Python
transcription of its estimator: all 10 scrolls 1.90 -> 1.44 mm (p = 0.0082), held-out 7 2.12 -> 1.46 mm (p = 0.076).

`vc3d/uncrush.patch` adds the step to VC3D itself, as an opt-in argument `uncrush` of `align_and_extract_umbilicus` and
an option `--uncrush` of `vc_gen_umbilicus`. With 0, the default, the output is unchanged bit for bit.

In C++ the stretch comes from the orientation tensor of the segment normals, not from the spread of the midpoints. The
normal tensor of a round section spanning half a turn or more is round, so such a section is left alone. A crushed
section's normals crowd across its flat sides, and the section is stretched back along that axis, with strength beta =
1.5 chosen on the 3 dev scrolls. The patch adds 3 unit tests.

`vc3d/build.sh` extracts the patched function verbatim from a villa checkout and compiles it against the OpenCV
stand-ins in `cvmock.hpp`, so the numbers below come from the patch's own code. It has not been built inside VC3D here.

Distance in mm to manual umbilici, 9 seeds per slice (numbers and per-slice values: `results/qa/umbilicus_vc3d_cpp.json`):

| Reference set | Slices | Unchanged | Midpoint version (alpha 0.5) | Normal version (beta 1.5, the patch) |
|---|---|---|---|---|
| Drobkov, 7 held-out scrolls | 56 | 2.22 | 1.79 (p = 0.0069) | **1.33** (35 better / 18 worse, p = 0.0074) |
| Drobkov, all 10 (3 were used to pick beta) | 80 | 2.00 | 1.68 (p = 0.0007) | 1.28 (44 / 30, p = 0.0147) |
| Challenge umbilici, 6 scrolls (pre-registered) | 48 | 0.85 | 0.69 (p = 0.0725) | 0.61 (20 / 19, p = 0.113) |

The challenge set does not reach significance with either version; the pre-registered pass needed p < 0.05.

On synthetic sections (20 seeds):
- A crushed 2.5:1 half section goes from 2.42 to 0.85 mm, and a crushed 4:1 section from 4.10 to 0.75 mm.
- A round half circle stays at 2.41 mm, where the midpoint version made it 4.58 mm.
```
bash vc3d/build.sh VILLA_DIR BUILD                         # villa with uncrush.patch applied
python vc3d/eval_drobkov.py OUT BUILD/vc3d_harness          # OUT from fetch_meta.py + umb_bench.py + umb_vc3d.py; uncrush 0 vs 1.5
python vc3d/list_challenge.py CH && python vc3d/meta_challenge.py CH && python vc3d/eval_challenge.py CH BUILD/vc3d_harness
```

## Downstream: does the axis change the spiral fit? (`results/qa/umbilicus_downstream_1447.json`)
The test band is PHerc1447 z 13100–14100, run with the kit's tracks-on spiral fit (10,000 steps). Only `umbilicus.json`
changes between arms. Each fit is scored on the organisers' labelled surfaces w058/w060 with `tools/surf_wind.py`.

| Axis | Error vs hand-made | Text px on the modal winding | Satisfied tracks |
|---|---|---|---|
| Hand-made (Drobkov) | 0 | 0.829 | 37.4 % |
| Hand-made, repeat run | 0 | 0.821 | 37.5 % |
| Un-crushed auto (beta 1.5), two runs | 0.57 mm | 0.819, 0.821 | 37.7 %, 37.7 % |
| Unchanged auto, two runs | 1.28 mm | 0.791, 0.787 | 35.1 %, 35.1 % |
| Hand-made + 300 voxels (sabotage) | 2.59 mm | 0.720 | 35.1 % |

The un-crushed axis fits as well as the hand-made axis. The unchanged axis loses about 2.3 points of satisfied tracks,
and the shifted axis loses on every measure. Repeat runs agree to 0.01 or better. This is one band of one scroll.
