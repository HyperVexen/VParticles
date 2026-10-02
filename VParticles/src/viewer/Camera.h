#pragma once

#include "Math3D.h"
#include <algorithm>

namespace vparticles::viewer {

class Camera {
public:
    Camera(const Vec3& target = {0.0f, 5.0f, 0.0f},
           float distance = 40.0f,
           float yaw = 45.0f,
           float pitch = 25.0f,
           float fovDegrees = 50.0f)
        : target_(target)
        , distance_(distance)
        , yaw_(yaw)
        , pitch_(pitch)
        , fovDegrees_(fovDegrees)
    {}

    void reset(const Vec3& target = {0.0f, 5.0f, 0.0f},
               float distance = 40.0f,
               float yaw = 45.0f,
               float pitch = 25.0f) {
        target_ = target;
        distance_ = distance;
        yaw_ = yaw;
        pitch_ = pitch;
    }

    void processOrbit(float dx, float dy, float sensitivity = 0.3f) {
        yaw_ += dx * sensitivity;
        pitch_ += dy * sensitivity;
        pitch_ = std::clamp(pitch_, -89.0f, 89.0f);
    }

    void processPan(float dx, float dy, float sensitivity = 0.04f) {
        Vec3 forward = (target_ - eyePosition()).normalized();
        Vec3 right = cross(forward, {0.0f, 1.0f, 0.0f}).normalized();
        Vec3 up = cross(right, forward).normalized();

        float scale = distance_ * sensitivity * 0.05f;
        target_ -= right * (dx * scale);
        target_ += up * (dy * scale);
    }

    void processZoom(float yoffset, float sensitivity = 2.0f) {
        distance_ -= yoffset * sensitivity * (distance_ * 0.08f);
        distance_ = std::clamp(distance_, 1.5f, 2000.0f);
    }

    Vec3 eyePosition() const {
        float yawRad = toRadians(yaw_);
        float pitchRad = toRadians(pitch_);
        float cosPitch = std::cos(pitchRad);

        float x = target_.x + distance_ * cosPitch * std::sin(yawRad);
        float y = target_.y + distance_ * std::sin(pitchRad);
        float z = target_.z + distance_ * cosPitch * std::cos(yawRad);

        return {x, y, z};
    }

    Mat4 viewMatrix() const {
        return Mat4::lookAt(eyePosition(), target_, {0.0f, 1.0f, 0.0f});
    }

    Mat4 projectionMatrix(float aspect, float nearZ = 0.2f, float farZ = 5000.0f) const {
        return Mat4::perspective(toRadians(fovDegrees_), aspect, nearZ, farZ);
    }

    Mat4 viewProjectionMatrix(float aspect, float nearZ = 0.2f, float farZ = 5000.0f) const {
        return projectionMatrix(aspect, nearZ, farZ) * viewMatrix();
    }

    const Vec3& target() const { return target_; }
    void setTarget(const Vec3& target) { target_ = target; }
    float distance() const { return distance_; }
    void setDistance(float d) { distance_ = std::clamp(d, 1.5f, 2000.0f); }
    float fovDegrees() const { return fovDegrees_; }

private:
    Vec3 target_;
    float distance_;
    float yaw_;
    float pitch_;
    float fovDegrees_;
};

} // namespace vparticles::viewer
