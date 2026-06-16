#pragma once

#include <cmath>
#include <cstring>

// Minimal 4x4 matrix math for camera MVP — no external dependencies.
// Column-major layout matching OpenGL expectations (mat[col][row]).
struct Mat4
{
    float m[16]; // column-major: m[col*4 + row]

    static Mat4 Identity()
    {
        Mat4 r;
        std::memset(r.m, 0, sizeof(r.m));
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    static Mat4 Perspective(float fovYRadians, float aspect, float nearPlane, float farPlane)
    {
        float tanHalfFov = std::tan(fovYRadians * 0.5f);
        Mat4 r;
        std::memset(r.m, 0, sizeof(r.m));
        r.m[0]  = 1.0f / (aspect * tanHalfFov);
        r.m[5]  = 1.0f / tanHalfFov;
        r.m[10] = -(farPlane + nearPlane) / (farPlane - nearPlane);
        r.m[11] = -1.0f;
        r.m[14] = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
        return r;
    }

    static Mat4 LookAt(float eyeX, float eyeY, float eyeZ,
                       float cenX, float cenY, float cenZ,
                       float upX,  float upY,  float upZ)
    {
        // Forward (from eye to center)
        float fx = cenX - eyeX, fy = cenY - eyeY, fz = cenZ - eyeZ;
        float fLen = std::sqrt(fx*fx + fy*fy + fz*fz);
        fx /= fLen; fy /= fLen; fz /= fLen;

        // Right = forward x up
        float rx = fy*upZ - fz*upY;
        float ry = fz*upX - fx*upZ;
        float rz = fx*upY - fy*upX;
        float rLen = std::sqrt(rx*rx + ry*ry + rz*rz);
        rx /= rLen; ry /= rLen; rz /= rLen;

        // True up = right x forward
        float ux = ry*fz - rz*fy;
        float uy = rz*fx - rx*fz;
        float uz = rx*fy - ry*fx;

        Mat4 r;
        std::memset(r.m, 0, sizeof(r.m));
        r.m[0]  = rx;  r.m[4]  = ry;  r.m[8]  = rz;  r.m[12] = -(rx*eyeX + ry*eyeY + rz*eyeZ);
        r.m[1]  = ux;  r.m[5]  = uy;  r.m[9]  = uz;  r.m[13] = -(ux*eyeX + uy*eyeY + uz*eyeZ);
        r.m[2]  = -fx; r.m[6]  = -fy; r.m[10] = -fz; r.m[14] =  (fx*eyeX + fy*eyeY + fz*eyeZ);
        r.m[3]  = 0;   r.m[7]  = 0;   r.m[11] = 0;   r.m[15] = 1.0f;
        return r;
    }

    // Matrix multiply: result = a * b (column-major)
    static Mat4 Multiply(const Mat4& a, const Mat4& b)
    {
        Mat4 r;
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
            {
                r.m[col*4 + row] =
                    a.m[0*4 + row] * b.m[col*4 + 0] +
                    a.m[1*4 + row] * b.m[col*4 + 1] +
                    a.m[2*4 + row] * b.m[col*4 + 2] +
                    a.m[3*4 + row] * b.m[col*4 + 3];
            }
        return r;
    }

    static Mat4 Translate(float tx, float ty, float tz)
    {
        Mat4 r = Identity();
        r.m[12] = tx;
        r.m[13] = ty;
        r.m[14] = tz;
        return r;
    }

    static Mat4 Scale(float sx, float sy, float sz)
    {
        Mat4 r;
        std::memset(r.m, 0, sizeof(r.m));
        r.m[0] = sx;
        r.m[5] = sy;
        r.m[10] = sz;
        r.m[15] = 1.0f;
        return r;
    }

    const float* Ptr() const { return m; }
};

// Simple orbital camera that orbits around a target point.
// Controlled by yaw/pitch angles and distance.
struct Camera
{
    // Orbit parameters
    float targetX = 640.0f;
    float targetY = 360.0f;
    float targetZ = 0.0f;

    float yaw   = 0.0f;    // horizontal angle (radians)
    float pitch  = 0.0f;   // vertical angle (radians), clamped to avoid gimbal lock
    float distance = 800.0f; // distance from target

    // Projection parameters
    float fovY      = 60.0f * 3.14159265f / 180.0f; // 60 degrees in radians
    float nearPlane = 1.0f;
    float farPlane  = 50000.0f;

    // Compute eye position from orbit parameters
    void GetEyePosition(float& ex, float& ey, float& ez) const
    {
        ex = targetX + distance * std::cos(pitch) * std::sin(yaw);
        ey = targetY + distance * std::sin(pitch);
        ez = targetZ + distance * std::cos(pitch) * std::cos(yaw);
    }

    Mat4 GetViewMatrix() const
    {
        float ex, ey, ez;
        GetEyePosition(ex, ey, ez);
        return Mat4::LookAt(ex, ey, ez, targetX, targetY, targetZ, 0.0f, 1.0f, 0.0f);
    }

    Mat4 GetProjectionMatrix(float aspectRatio) const
    {
        return Mat4::Perspective(fovY, aspectRatio, nearPlane, farPlane);
    }

    Mat4 GetVPMatrix(float aspectRatio) const
    {
        return Mat4::Multiply(GetProjectionMatrix(aspectRatio), GetViewMatrix());
    }

    // Clamp pitch to avoid flipping
    void ClampPitch()
    {
        const float limit = 89.0f * 3.14159265f / 180.0f;
        if (pitch >  limit) pitch =  limit;
        if (pitch < -limit) pitch = -limit;
    }

    void Pan(float dx, float dy)
    {
        // Camera local Right vector
        float rx = std::cos(yaw);
        float ry = 0.0f;
        float rz = -std::sin(yaw);

        // Camera local Up vector
        float ux = -std::sin(yaw) * std::sin(pitch);
        float uy = std::cos(pitch);
        float uz = -std::cos(yaw) * std::sin(pitch);

        float factor = 0.002f * distance;

        targetX -= rx * dx * factor + ux * dy * factor;
        targetY -= ry * dx * factor + uy * dy * factor;
        targetZ -= rz * dx * factor + uz * dy * factor;
    }
};
