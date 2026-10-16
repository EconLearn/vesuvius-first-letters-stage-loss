#!/bin/bash
# Build the harness: extract align_and_extract_umbilicus VERBATIM from a villa checkout's normalgridtools.cpp (with
# uncrush.patch applied) and compile it against the OpenCV stand-ins in cvmock.hpp.
# usage: build.sh VILLA_DIR OUT_DIR      (needs g++ with C++17)
set -euo pipefail
V=$1; O=$2; D=$(cd "$(dirname "$0")" && pwd); mkdir -p "$O"
S=$V/volume-cartographer/core/src/normalgridtools.cpp
a=$(grep -n "^cv::Vec2f align_and_extract_umbilicus" "$S" | cut -d: -f1); b=$(grep -n "^void align_and_filter_segments" "$S" | cut -d: -f1)
sed -n "${a},$((b-1))p" "$S" | sed 's/^cv::Vec2f align_and_extract_umbilicus(/cv::Vec2f vc_align_and_extract_umbilicus(/' > "$O/extracted.inc"
grep -q "float uncrush" "$O/extracted.inc" || { echo "normalgridtools.cpp has no uncrush argument: apply uncrush.patch first"; exit 1; }
g++ -O2 -std=c++17 -Wall -Wextra -I"$O" -I"$D" -o "$O/vc3d_harness" "$D/main.cpp"
g++ -O2 -std=c++17 -Wall -Wextra -I"$O" -I"$D" -o "$O/vc3d_synth" "$D/synth.cpp"
echo "built $O/vc3d_harness $O/vc3d_synth"
