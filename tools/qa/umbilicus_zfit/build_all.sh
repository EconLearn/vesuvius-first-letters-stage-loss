#!/bin/zsh
# Builds the drivers and the test binary from the villa checkout (OpenCV stand-in cvshim/, nlohmann header in inc/).
set -e
H=${0:A:h}; V=${1:-$H/../../vesuvius_pr2123/villa}; VC=$V/volume-cartographer
F=(-std=c++23 -O2 -w -I$H/cvshim -I$H/inc -I$VC/core/include -I$VC/utils/include -I$V/vc_reader/include -I$V/vc_object_store/include)
SRC=($VC/core/src/normalgridtools.cpp $VC/core/src/GridStore.cpp $V/vc_object_store/src/MemMap.cpp)
clang++ $F $SRC $H/zfitdrv.cpp -o $H/zfitdrv
clang++ $F $SRC $H/tolprobe.cpp -o $H/tolprobe
clang++ $F -I$VC/core/test/doctest_compat -I$VC/core/test -DVC_TEST_FIXTURES_DIR="\"$VC/core/test/data\"" $SRC $VC/core/test/test_normalgridtools.cpp -o $H/test_normalgridtools
echo built
