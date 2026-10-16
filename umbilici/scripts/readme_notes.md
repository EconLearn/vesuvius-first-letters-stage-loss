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
