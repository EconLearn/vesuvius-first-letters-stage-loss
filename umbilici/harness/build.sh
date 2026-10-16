# Build the two estimator binaries from these sources (default output dir: /tmp/umbkit_build; pass another dir as $1).
# vc3d_harness   ALPHA SEED file.grid...                 (ALPHA 0 = VC3D unchanged, 0.5 = midpoint un-crush)
# vc3d_harness_n BETA SEED file.grid... with UNCRUSH_N=1 (normal un-crush; without UNCRUSH_N it equals vc3d_harness)
set -e
OUT=${1:-/tmp/umbkit_build}; mkdir -p "$OUT"; cd "$(dirname "$0")"
g++ -O2 -std=c++17 -o "$OUT/vc3d_harness" main.cpp
g++ -O2 -std=c++17 -o "$OUT/vc3d_harness_n" main_n.cpp
sha256sum "$OUT/vc3d_harness" "$OUT/vc3d_harness_n"
