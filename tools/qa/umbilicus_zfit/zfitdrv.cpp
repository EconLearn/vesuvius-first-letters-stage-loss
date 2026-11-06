// Reads "z x y" lines on stdin and the query z from argv[2..]; prints the fit_umbilicus_along_z output, one "x y" per query.
// usage: zfitdrv HALF_WIDTH < estimates.txt  (query = every distinct z of the input, in order)
#include "vc/core/util/normalgridtools.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace vc::core::util;
int main(int argc, char** argv) {
    const double H = std::atof(argv[1]);
    std::vector<cv::Vec3d> e; std::vector<double> q; double z, x, y;
    while (std::cin >> z >> x >> y) { e.emplace_back(z, x, y); if (q.empty() || q.back() != z) q.push_back(z); }
    for (const auto& v : fit_umbilicus_along_z(e, q, H)) std::printf("%.9f %.9f\n", v[0], v[1]);
}
