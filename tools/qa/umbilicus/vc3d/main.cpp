// usage: harness ALPHA SEED file.grid [...]   -> one line per file: x y   (the patched estimator's output)
#include "cvmock.hpp"
#include <fstream>
#include <cstring>
#include <sstream>
#include <iterator>
#include <algorithm>
#include "extracted.inc"
static uint32_t be(const unsigned char* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
int main(int argc, char** argv) {
    float alpha = std::atof(argv[1]); uint32_t seed = std::strtoul(argv[2], nullptr, 10);
    for (int a = 3; a < argc; ++a) {
        std::ifstream f(argv[a], std::ios::binary); std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), {});
        if (b.size() < 52 || std::memcmp(b.data(), "VCGS", 4)) { std::cout << "nan nan\n"; continue; }
        uint32_t h[13]; for (int i = 0; i < 13; ++i) h[i] = be(&b[4 * i]);
        GridStore gs; gs.sz.width = h[4]; gs.sz.height = h[5]; size_t o = h[10];
        for (uint32_t p = 0; p < h[8]; ++p) {
            int x = be(&b[o]), y = be(&b[o + 4]); uint32_t n = be(&b[o + 8]); o += 12;
            auto path = std::make_shared<std::vector<cv::Point>>(); path->emplace_back(x, y);
            for (uint32_t k = 0; k < n / 2; ++k) { x += (int8_t)b[o + 2 * k]; y += (int8_t)b[o + 2 * k + 1]; path->emplace_back(x, y); }
            o += n; gs.paths.push_back(path);
        }
        std::streambuf* old = std::cout.rdbuf(); std::ostringstream sink; std::cout.rdbuf(sink.rdbuf());
        cv::Vec2f u = vc_align_and_extract_umbilicus(gs, seed, alpha);
        std::cout.rdbuf(old); std::cout << u[0] << " " << u[1] << "\n";
    }
}
