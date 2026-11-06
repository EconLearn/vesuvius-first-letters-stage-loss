// 20-seed spread of the outvoting case, to set the test tolerance (same construction as the test).
#include "vc/core/util/normalgridtools.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace vc::core::util;
cv::Vec2d axis(double z) { return cv::Vec2d(3000.0 + 0.03 * z, 2500.0 - 0.01 * z); }
int main() {
    std::vector<double> rob, pl, raw;
    for (unsigned seed = 1; seed <= 20; ++seed) {
        std::mt19937 rng(seed); std::normal_distribution<double> noise(0.0, 20.0);
        std::uniform_real_distribution<double> angle(0.0, 6.283185307179586);
        std::vector<cv::Vec3d> e; std::vector<double> sl;
        for (int i = 0; i < 60; ++i) { double z = 256.0 * i; cv::Vec2d p = axis(z) + cv::Vec2d(noise(rng), noise(rng));
            if (i % 7 == 3) { double a = angle(rng); p += cv::Vec2d(400 * std::cos(a), 400 * std::sin(a)); }
            e.emplace_back(z, p[0], p[1]); sl.push_back(z); }
        auto w = [&](const std::vector<cv::Vec2d>& f) { double m = 0; for (size_t i = 0; i < sl.size(); ++i) { auto d = f[i] - axis(sl[i]); m = std::max(m, std::hypot(d[0], d[1])); } return m; };
        rob.push_back(w(fit_umbilicus_along_z(e, sl, 2000.0))); pl.push_back(w(fit_umbilicus_along_z(e, sl, 2000.0, 0)));
        std::printf("seed %u robust %.2f plain %.2f\n", seed, rob.back(), pl.back());
    }
    std::sort(rob.begin(), rob.end()); std::sort(pl.begin(), pl.end());
    std::printf("robust max %.2f median %.2f | plain min %.2f median %.2f\n", rob.back(), rob[10], pl.front(), pl[10]);
}
