// Real-scroll driver: load each published normal grid with VC3D's own GridStore loader and print the
// umbilicus estimate of the linked normalgridtools.cpp.  usage: evaldrv BETA SEED file.grid [...] -> "x y" per file
#include "vc/core/util/normalgridtools.hpp"
#include "vc/core/util/GridStore.hpp"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
using namespace vc::core::util;
int main(int argc, char** argv) {
    const float beta = std::atof(argv[1]);
    const std::uint32_t seed = std::strtoul(argv[2], nullptr, 10);
    for (int a = 3; a < argc; ++a) {
        GridStore gs{std::string(argv[a])};
        std::streambuf* old = std::cout.rdbuf(); std::ostringstream sink; std::cout.rdbuf(sink.rdbuf());
        cv::Vec2f u = align_and_extract_umbilicus(gs, seed, beta);
        std::cout.rdbuf(old);
        std::cout << u[0] << " " << u[1] << "\n";
    }
}
