#pragma once

#include <cmath>
#include <cstring>

namespace vparticles::viewer {

constexpr float kPi = 3.14159265358979323846f;

inline float toRadians(float degrees) {
    return degrees * (kPi / 180.0f);
}

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float lengthSq() const { return x * x + y * y + z * z; }
    float length() const { return std::sqrt(lengthSq()); }

    Vec3 normalized() const {
        float len = length();
        if (len > 1e-6f) {
            float inv = 1.0f / len;
            return {x * inv, y * inv, z * inv};
        }
        return {0.0f, 0.0f, 0.0f};
    }
};

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// Column-major 4x4 matrix, directly compatible with OpenGL glUniformMatrix4fv(..., GL_FALSE, ...)
struct Mat4 {
    float m[16] = {0.0f};

    static Mat4 identity() {
        Mat4 res;
        res.m[0]  = 1.0f;
        res.m[5]  = 1.0f;
        res.m[10] = 1.0f;
        res.m[15] = 1.0f;
        return res;
    }

    const float* data() const { return m; }
    float* data() { return m; }

    Mat4 operator*(const Mat4& rhs) const {
        Mat4 out;
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    sum += m[k * 4 + r] * rhs.m[c * 4 + k];
                }
                out.m[c * 4 + r] = sum;
            }
        }
        return out;
    }

    static Mat4 perspective(float fovRad, float aspect, float nearZ, float farZ) {
        Mat4 res;
        float tanHalfFov = std::tan(fovRad * 0.5f);
        res.m[0]  = 1.0f / (aspect * tanHalfFov);
        res.m[5]  = 1.0f / tanHalfFov;
        res.m[10] = -(farZ + nearZ) / (farZ - nearZ);
        res.m[11] = -1.0f;
        res.m[14] = -(2.0f * farZ * nearZ) / (farZ - nearZ);
        res.m[15] = 0.0f;
        return res;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
        Vec3 f = (target - eye).normalized();
        Vec3 s = cross(f, up).normalized();
        Vec3 u = cross(s, f);

        Mat4 res = Mat4::identity();
        res.m[0]  = s.x;
        res.m[4]  = s.y;
        res.m[8]  = s.z;

        res.m[1]  = u.x;
        res.m[5]  = u.y;
        res.m[9]  = u.z;

        res.m[2]  = -f.x;
        res.m[6]  = -f.y;
        res.m[10] = -f.z;

        res.m[12] = -dot(s, eye);
        res.m[13] = -dot(u, eye);
        res.m[14] =  dot(f, eye);
        return res;
    }
};

} // namespace vparticles::viewer
