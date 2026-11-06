// Minimal stand-in for the parts of OpenCV 4 that normalgridtools.cpp, GridStore.cpp and
// test_normalgridtools.cpp use. It exists only so that the real sources can be compiled and
// tested on a Mac without OpenCV installed (no brew opencv: ~60 missing dependencies, several
// GB, on a disk with 28 GB free). The arithmetic of the functions the estimator depends on
// follows OpenCV 4:
//   cv::norm(Vec)            sqrt of the float-accumulated sum of squares (normL2Sqr<float,float>)
//   cv::normalize(src, dst)  the InputArray path: scale = 1 / ||src|| computed in double
//                            (0 if ||src|| <= DBL_EPSILON), then dst = src * (float)scale in float
//   Point_ <-> Vec, Rect &, Point / int as in core/types.hpp and core/matx.hpp
// Drawing and image writing are no-ops; they are only used by the visualisation helpers.
#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#ifndef CV_8UC3
#define CV_8UC3 16
#endif

namespace cv {

inline int cvRoundImpl(double v) { return (int)std::lrint(v); }

template <typename T> inline T saturate_cast(double v) { return (T)v; }
template <> inline int saturate_cast<int>(double v) { return cvRoundImpl(v); }
template <> inline float saturate_cast<float>(double v) { return (float)v; }

template <typename T, int cn> struct Vec {
    T val[cn];
    Vec() { for (int i = 0; i < cn; ++i) val[i] = T(0); }
    Vec(T v0, T v1) { static_assert(cn == 2, "Vec2 only"); val[0] = v0; val[1] = v1; }
    Vec(T v0, T v1, T v2) { static_assert(cn == 3, "Vec3 only"); val[0] = v0; val[1] = v1; val[2] = v2; }
    T& operator[](int i) { return val[i]; }
    const T& operator[](int i) const { return val[i]; }
    T dot(const Vec& o) const { T s = 0; for (int i = 0; i < cn; ++i) s += val[i] * o.val[i]; return s; }
    Vec& operator+=(const Vec& o) { for (int i = 0; i < cn; ++i) val[i] += o.val[i]; return *this; }
    Vec& operator-=(const Vec& o) { for (int i = 0; i < cn; ++i) val[i] -= o.val[i]; return *this; }
    Vec& operator*=(T s) { for (int i = 0; i < cn; ++i) val[i] = saturate_cast<T>(val[i] * s); return *this; }
    Vec& operator/=(T s) {   // OpenCV (matx.hpp), Vec<float> /= float: multiply by 1.f / alpha
        const T inv = T(1) / s;
        for (int i = 0; i < cn; ++i) val[i] *= inv;
        return *this;
    }
};
template <typename T, int cn> inline Vec<T, cn> operator+(Vec<T, cn> a, const Vec<T, cn>& b) { a += b; return a; }
template <typename T, int cn> inline Vec<T, cn> operator-(Vec<T, cn> a, const Vec<T, cn>& b) { a -= b; return a; }
template <typename T, int cn> inline Vec<T, cn> operator-(const Vec<T, cn>& a) { Vec<T, cn> r; for (int i = 0; i < cn; ++i) r.val[i] = -a.val[i]; return r; }
template <typename T, int cn> inline Vec<T, cn> operator*(Vec<T, cn> a, double s) { for (int i = 0; i < cn; ++i) a.val[i] = saturate_cast<T>(a.val[i] * s); return a; }
template <typename T, int cn> inline Vec<T, cn> operator*(double s, Vec<T, cn> a) { return a * s; }
template <typename T, int cn> inline Vec<T, cn> operator*(Vec<T, cn> a, float s) { for (int i = 0; i < cn; ++i) a.val[i] = saturate_cast<T>(a.val[i] * s); return a; }
template <typename T, int cn> inline Vec<T, cn> operator*(Vec<T, cn> a, int s) { for (int i = 0; i < cn; ++i) a.val[i] = saturate_cast<T>(a.val[i] * s); return a; }
template <typename T, int cn> inline Vec<T, cn> operator/(Vec<T, cn> a, T s) { a /= s; return a; }
template <typename T, int cn> inline bool operator==(const Vec<T, cn>& a, const Vec<T, cn>& b) { for (int i = 0; i < cn; ++i) if (!(a.val[i] == b.val[i])) return false; return true; }
template <typename T, int cn> inline bool operator!=(const Vec<T, cn>& a, const Vec<T, cn>& b) { return !(a == b); }
template <typename T, int cn> inline std::ostream& operator<<(std::ostream& os, const Vec<T, cn>& v) {
    os << "["; for (int i = 0; i < cn; ++i) { if (i) os << ", "; os << v.val[i]; } return os << "]";
}

typedef Vec<float, 2> Vec2f;
typedef Vec<double, 2> Vec2d;
typedef Vec<int, 2> Vec2i;
typedef Vec<double, 3> Vec3d;

// OpenCV: norm(const Vec<_Tp,cn>&) = std::sqrt(normL2Sqr<_Tp, DataType<_Tp>::work_type>(v.val, cn)),
// work_type of float is float.
template <typename T, int cn> inline double norm(const Vec<T, cn>& v) {
    T s = 0; for (int i = 0; i < cn; ++i) s += v.val[i] * v.val[i];
    return std::sqrt(s);
}
// OpenCV: normalize(InputArray, InputOutputArray) with NORM_L2: norm in double, then convertTo with
// float alpha.
template <typename T, int cn> inline void normalize(const Vec<T, cn>& src, Vec<T, cn>& dst) {
    double s = 0; for (int i = 0; i < cn; ++i) s += (double)src.val[i] * (double)src.val[i];
    s = std::sqrt(s);
    const double scale = s > DBL_EPSILON ? 1.0 / s : 0.0;
    const float a = (float)scale;
    Vec<T, cn> r;
    for (int i = 0; i < cn; ++i) r.val[i] = (T)(src.val[i] * a + 0.0f);
    dst = r;
}
template <typename T, int cn> inline Vec<T, cn> normalize(const Vec<T, cn>& v) {
    const double nv = norm(v);
    return v * (nv ? 1. / nv : 0.);
}

template <typename T> struct Point_ {
    T x, y;
    Point_() : x(0), y(0) {}
    Point_(T x_, T y_) : x(x_), y(y_) {}
    Point_(const Vec<T, 2>& v) : x(v.val[0]), y(v.val[1]) {}
    template <typename T2> operator Point_<T2>() const { return Point_<T2>(saturate_cast<T2>((double)x), saturate_cast<T2>((double)y)); }
    operator Vec<T, 2>() const { return Vec<T, 2>(x, y); }
    T dot(const Point_& p) const { return saturate_cast<T>((double)x * p.x + (double)y * p.y); }
    double ddot(const Point_& p) const { return (double)x * p.x + (double)y * p.y; }
    Point_& operator+=(const Point_& b) { x += b.x; y += b.y; return *this; }
    Point_& operator-=(const Point_& b) { x -= b.x; y -= b.y; return *this; }
};
template <typename T> inline Point_<T> operator+(Point_<T> a, const Point_<T>& b) { a += b; return a; }
template <typename T> inline Point_<T> operator-(Point_<T> a, const Point_<T>& b) { a -= b; return a; }
template <typename T> inline Point_<T> operator-(const Point_<T>& a) { return Point_<T>(-a.x, -a.y); }
template <typename T> inline Point_<T> operator*(const Point_<T>& a, int b) { return Point_<T>(saturate_cast<T>((double)(a.x * b)), saturate_cast<T>((double)(a.y * b))); }
template <typename T> inline Point_<T> operator*(const Point_<T>& a, float b) { return Point_<T>(saturate_cast<T>(a.x * b), saturate_cast<T>(a.y * b)); }
template <typename T> inline Point_<T> operator*(const Point_<T>& a, double b) { return Point_<T>(saturate_cast<T>(a.x * b), saturate_cast<T>(a.y * b)); }
template <typename T> inline Point_<T> operator/(const Point_<T>& a, int b) { return Point_<T>(saturate_cast<T>((double)(a.x / b)), saturate_cast<T>((double)(a.y / b))); }
template <typename T> inline Point_<T> operator/(const Point_<T>& a, float b) { return Point_<T>(saturate_cast<T>(a.x / b), saturate_cast<T>(a.y / b)); }
template <typename T> inline bool operator==(const Point_<T>& a, const Point_<T>& b) { return a.x == b.x && a.y == b.y; }
template <typename T> inline bool operator!=(const Point_<T>& a, const Point_<T>& b) { return !(a == b); }
template <typename T> inline double norm(const Point_<T>& p) { return std::sqrt((double)p.x * p.x + (double)p.y * p.y); }
template <typename T> inline std::ostream& operator<<(std::ostream& os, const Point_<T>& p) { return os << "[" << p.x << ", " << p.y << "]"; }

typedef Point_<int> Point2i;
typedef Point_<int> Point;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;

template <typename T> struct Size_ {
    T width, height;
    Size_() : width(0), height(0) {}
    Size_(T w, T h) : width(w), height(h) {}
    T area() const { return width * height; }
    bool empty() const { return width <= 0 || height <= 0; }
};
template <typename T> inline bool operator==(const Size_<T>& a, const Size_<T>& b) { return a.width == b.width && a.height == b.height; }
template <typename T> inline bool operator!=(const Size_<T>& a, const Size_<T>& b) { return !(a == b); }
template <typename T> inline std::ostream& operator<<(std::ostream& os, const Size_<T>& s) { return os << "[" << s.width << " x " << s.height << "]"; }
typedef Size_<int> Size;

template <typename T> struct Rect_ {
    T x, y, width, height;
    Rect_() : x(0), y(0), width(0), height(0) {}
    Rect_(T x_, T y_, T w, T h) : x(x_), y(y_), width(w), height(h) {}
    Point_<T> tl() const { return Point_<T>(x, y); }
    Point_<T> br() const { return Point_<T>(x + width, y + height); }
    Size_<T> size() const { return Size_<T>(width, height); }
    T area() const { return width * height; }
    bool empty() const { return width <= 0 || height <= 0; }
    bool contains(const Point_<T>& p) const { return x <= p.x && p.x < x + width && y <= p.y && p.y < y + height; }
};
template <typename T> inline Rect_<T> operator&(Rect_<T> a, const Rect_<T>& b) {
    if (a.empty() || b.empty()) return Rect_<T>();
    T x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1; a.y = y1;
    if (a.width <= 0 || a.height <= 0) a = Rect_<T>();
    return a;
}
template <typename T> inline bool operator==(const Rect_<T>& a, const Rect_<T>& b) { return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height; }
template <typename T> inline std::ostream& operator<<(std::ostream& os, const Rect_<T>& r) { return os << "[" << r.width << " x " << r.height << " from (" << r.x << ", " << r.y << ")]"; }
typedef Rect_<int> Rect;

struct Scalar {
    double val[4];
    Scalar(double a = 0, double b = 0, double c = 0, double d = 0) { val[0] = a; val[1] = b; val[2] = c; val[3] = d; }
};

enum { LINE_4 = 4, LINE_8 = 8, LINE_AA = 16 };

// A Mat that only knows its shape. Drawing into it is a no-op.
class Mat {
public:
    int rows = 0, cols = 0, type_ = 0;
    Mat() = default;
    Mat(int r, int c, int t) : rows(r), cols(c), type_(t) {}
    static Mat zeros(int r, int c, int t) { return Mat(r, c, t); }
    bool empty() const { return rows <= 0 || cols <= 0; }
    Size size() const { return Size(cols, rows); }
    int type() const { return type_; }
};

template <typename P> inline void line(Mat&, P, P, const Scalar&, int = 1, int = LINE_8, int = 0) {}
template <typename P> inline void circle(Mat&, P, int, const Scalar&, int = 1, int = LINE_8, int = 0) {}
template <typename P> inline void arrowedLine(Mat&, P, P, const Scalar&, int = 1, int = 8, int = 0, double = 0.1) {}
inline bool imwrite(const std::string&, const Mat&, const std::vector<int>& = std::vector<int>()) { return true; }

} // namespace cv

inline int cvRound(double v) { return cv::cvRoundImpl(v); }
inline int cvRound(float v) { return cv::cvRoundImpl((double)v); }
