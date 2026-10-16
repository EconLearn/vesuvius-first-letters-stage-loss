// Minimal stand-ins for the few OpenCV types align_and_extract_umbilicus uses, so that the function's
// own source text (extracted from normalgridtools.cpp) compiles here without OpenCV.
#pragma once
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <vector>
namespace cv {
struct Point { int x = 0, y = 0; Point() {} Point(int a, int b) : x(a), y(b) {} };
struct Point2f { float x = 0, y = 0; Point2f() {} Point2f(float a, float b) : x(a), y(b) {} };
struct Size { int width = 0, height = 0; };
struct Vec2f {
    float v[2] = {0, 0};
    Vec2f() {}
    Vec2f(float a, float b) { v[0] = a; v[1] = b; }
    explicit Vec2f(const Point2f& p) { v[0] = p.x; v[1] = p.y; }
    float& operator[](int i) { return v[i]; }
    const float& operator[](int i) const { return v[i]; }
    float dot(const Vec2f& o) const { return (float)((double)v[0] * o.v[0] + (double)v[1] * o.v[1]); }
    Vec2f operator-(const Vec2f& o) const { return Vec2f(v[0] - o.v[0], v[1] - o.v[1]); }
    Vec2f operator+(const Vec2f& o) const { return Vec2f(v[0] + o.v[0], v[1] + o.v[1]); }
    Vec2f& operator/=(float s) { v[0] /= s; v[1] /= s; return *this; }
};
inline double norm(const Vec2f& a) { return std::sqrt((double)a[0] * a[0] + (double)a[1] * a[1]); }
inline void normalize(const Vec2f& src, Vec2f& dst) {
    double n = norm(src); double s = n > 0 ? 1.0 / n : 0.0;
    dst = Vec2f((float)(src[0] * s), (float)(src[1] * s));
}
inline std::ostream& operator<<(std::ostream& o, const Vec2f& a) { return o << "[" << a[0] << ", " << a[1] << "]"; }
}
struct GridStore {
    std::vector<std::shared_ptr<std::vector<cv::Point>>> paths; cv::Size sz;
    const std::vector<std::shared_ptr<std::vector<cv::Point>>>& get_all() const { return paths; }
    cv::Size size() const { return sz; }
};
