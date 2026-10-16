# Downstream test: does the umbilicus change the spiral fit?

The same band is fitted several times with the official spiral fit (`spiral-fitting/fit_spiral.py`, through the cloud
kit's `fit_tag`: 10,000 steps, tracks on, track loss from step 7,000). Only `umbilicus.json` changes between fits. Each
fit is then scored in two ways:
- winding consistency on the organisers' labelled surfaces, with `tools/surf_wind.py` (the fraction of text pixels on
  one fitted winding);
- the fit's own track satisfaction.

1. Build the arms:
   ```
   python make_arms.py UMB_DIR HARNESS OUT PHerc1447 12900 14300 b1
   ```
   UMB_DIR is the benchmark directory of `../fetch_meta.py`; HARNESS comes from `../vc3d/build.sh`.
2. For each arm, make a dataset directory with the band's `lasagna_inputs/`, `tracks/` and `spiral-scroll.json`, plus
   the arm's `umbilicus.json`. Run `fit_tag TAG DATASET Z0 Z1 10000 7000` (tools/cloud/jobs/lib.sh).
3. Score each fit:
   ```
   python ../../../surf_wind.py CKPT DATASET/umbilicus.json ORG/w058,ORG/w060,ORG/w062 wind_ARM.json --stride 4
   ```
4. Collect the scores:
   ```
   python score_arms.py out.json hand=wind_hand.json,fit_hand.log unc=...
   ```

Results for PHerc1447 band 1 (z 13100–14100) are in `results/qa/umbilicus_downstream_1447.json`.
