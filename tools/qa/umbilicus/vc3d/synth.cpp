// usage: synth SQUASH FRAC THIN ALPHA SEED...  -> error in mm of the patched estimator on a synthetic section:
// concentric ellipses of semi-axes r and r / SQUASH (r 300..3500 step 80), centre (4000, 4000), points ~12 units apart
// on the uncrushed circle, FRAC of each ellipse kept, segments with x < centre thinned 1:THIN. SQUASH 1 is the
// existing test's addSyntheticSection exactly.
#include "cvmock.hpp"
#include <algorithm>
#include <sstream>
inline int cvRound(double v) { return (int)std::lrint(v); }
#include "extracted.inc"
constexpr double kPi = 3.14159265358979323846, kUnitMm = 0.009362, kCentreX = 4000.0, kCentreY = 4000.0;
constexpr int kSide = 8000;
void addCrushedSection(GridStore& gs, double squash, double frac, int thin_ratio, std::uint32_t thin_seed)
{
    std::mt19937 rng(thin_seed);
    std::uniform_int_distribution<int> keep(1, std::max(1, thin_ratio));
    for (double r = 300.0; r <= 3500.0; r += 80.0) {
        const int n = std::max(8, (int)(2.0 * kPi * r / 12.0));
        const int m = std::max(3, (int)(n * frac));
        std::vector<cv::Point> pts;
        pts.reserve(m);
        for (int i = 0; i < m; ++i) {
            const double th = frac >= 1.0 ? 2.0 * kPi * i / m : 2.0 * kPi * frac * i / (m - 1);
            pts.emplace_back(cvRound(kCentreX + r * std::cos(th)), cvRound(kCentreY + r / squash * std::sin(th)));
        }
        for (int i = 0; i + 1 < m; ++i) {
            const double mid_x = 0.5 * (pts[i].x + pts[i + 1].x);
            if (thin_ratio > 1 && mid_x < kCentreX && keep(rng) != 1) continue;
            gs.paths.push_back(std::make_shared<std::vector<cv::Point>>(std::vector<cv::Point>{pts[i], pts[i + 1]}));
        }
    }
}
int main(int argc, char** argv) {
    double squash = std::atof(argv[1]), frac = std::atof(argv[2]); int thin = std::atoi(argv[3]); float alpha = std::atof(argv[4]);
    GridStore gs; gs.sz.width = kSide; gs.sz.height = kSide; addCrushedSection(gs, squash, frac, thin, 3u);
    for (int a = 5; a < argc; ++a) {
        std::streambuf* old = std::cout.rdbuf(); std::ostringstream sink; std::cout.rdbuf(sink.rdbuf());
        cv::Vec2f u = vc_align_and_extract_umbilicus(gs, (uint32_t)std::atoi(argv[a]), alpha);
        std::cout.rdbuf(old);
        std::cout << std::hypot(u[0] - kCentreX, u[1] - kCentreY) * kUnitMm << " " << u[0] << " " << u[1] << "\n";
    }
}
