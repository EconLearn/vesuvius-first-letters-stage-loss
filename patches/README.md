# Patches

## `v8in-stitching-normalise-by-blend-weight.patch`

- **Target.** `ink8um/inference.py` of `YoussefMoNader/ink-8um-v8in` at revision
  `d89166b41a3f5fad7749b3d7c0fdd1bd3695d844` (file sha256 `bca4a2ba12e46f65df810b4f54d37a8b42ec9923e56f2a3c6acb3e3ffd984f38`).
  The upstream code is MIT-licensed.
- **The bug.** `predict_stack` accumulates `numerator += p * blend` but `count += 1`, then returns `numerator / count`.
  So the output equals the blend-weighted mean multiplied by W = Σblend / count. W is a data-independent grid with
  the 21-px tile stride. Inside canvases it ranges from 0.611 to 0.89; at coverage edges it reaches 0.38–0.94.
- **The patch** changes one line (`count += blend`) and the docstring line that describes the stitching.
- **Effect** (paired, PHerc1447 w058+w060, 4.16M labelled px):

  | Change | ΔAUC [95% CI] |
  |---|---|
  | correction on the released map | **+0.0008 [+0.0007, +0.0009]** |
  | sabotage: W shifted half a stride | −0.0006 [−0.0011, −0.0002] |

  The point is exactness and appearance: the grid shows up as lines on contrast-stretched low-signal crops.
- **Apply** from the model directory:

  ```
  patch -p1 < v8in-stitching-normalise-by-blend-weight.patch
  ```

- **Already-released maps** can be corrected without re-running the model: divide by W, rebuilt from the same tile
  layout. See `tools/g3_score.py`, functions `wfield` and `wfield_released`.
