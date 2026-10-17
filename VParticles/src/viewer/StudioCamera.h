#pragma once

#include "Math3D.h"
#include <algorithm>
#include <cmath>

namespace vparticles::viewer {

class StudioCamera {
public:
    StudioCamera(const Vec3& target = {0.0f, 0.35f, 0.0f},
                 float distance = 2.5f,
                 float yaw = 25.0f,
                 float pitch = 22.0f,
                 float fovDegrees = 50.0f)
        : target_(target)
        , distance_(distance)
        , yaw_(yaw)
        , pitch_(pitch)
        , fovDegrees_(fovDegrees)
    {}

    void reset(const Vec3& target = {0.0f, 0.35f, 0.0f},
               float distance = 2.5f,
               float yaw = 25.0f,
               float pitch = 22.0f) {
        target_ = target;
        distance_ = distance;
        yaw_ = yaw;
        pitch_ = pitch;
    }

    void processOrbit(float dx, float dy, float sensitivity = 0.28f) {
        yaw_ += dx * sensitivity;
        pitch_ += dy * sensitivity;
        pitch_ = std::clamp(pitch_, -89.5f, 89.5f);
    }

    void processPan(float dx, float dy, float sensitivity = 0.003f) {
        Vec3 fwd = forward();
        Vec3 rgt = right();
        Vec3 upVec = up();

        float scale = distance_ * sensitivity;
        target_ -= rgt * (dx * scale);
        target_ += upVec * (dy * scale);
    }

    void processZoom(float yoffset, float sensitivity = 0.15f) {
        distance_ -= yoffset * (distance_ * sensitivity);
        distance_ = std::clamp(distance_, 0.05f, 1000.0f);
    }

    // JangaFX Free-Fly Mode: update position from WASD/QE movement in fly mode
    void processFlyMovement(float fwdMove, float rightMove, float upMove, float dt, bool shiftFast) {
        float speed = flySpeed_ * (shiftFast ? 3.0f : 1.0f);
        Vec3 eye = eyePosition();
        Vec3 fwd = forward();
        Vec3 rgt = right();
        Vec3 upVec = {0.0f, 1.0f, 0.0f};

        Vec3 delta = (fwd * fwdMove + rgt * rightMove + upVec * upMove) * (speed * dt);
        target_ += delta;
    }

    void adjustFlySpeed(float delta) {
        flySpeed_ *= (delta > 0.0f ? 1.25f : 0.8f);
        flySpeed_ = std::clamp(flySpeed_, 0.1f, 50.0f);
    }

    // Direction vectors
    Vec3 forward() const {
        float yawRad = toRadians(yaw_);
        float pitchRad = toRadians(pitch_);
        float cosPitch = std::cos(pitchRad);
        // Looking from eye towards target
        return Vec3(-cosPitch * std::sin(yawRad),
                    -std::sin(pitchRad),
                    -cosPitch * std::cos(yawRad)).normalized();
    }

    Vec3 right() const {
        return cross(forward(), {0.0f, 1.0f, 0.0f}).normalized();
    }

    Vec3 up() const {
        return cross(right(), forward()).normalized();
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

    Mat4 projectionMatrix(float aspect, float nearZ = 0.02f, float farZ = 2000.0f) const {
        return Mat4::perspective(toRadians(fovDegrees_), aspect, nearZ, farZ);
    }

    Mat4 viewProjectionMatrix(float aspect, float nearZ = 0.02f, float farZ = 2000.0f) const {
        return projectionMatrix(aspect, nearZ, farZ) * viewMatrix();
    }

    const Vec3& target() const { return target_; }
    void setTarget(const Vec3& t) { target_ = t; }
    float distance() const { return distance_; }
    void setDistance(float d) { distance_ = std::clamp(d, 0.05f, 1000.0f); }
    float yaw() const { return yaw_; }
    float pitch() const { return pitch_; }
    float fovDegrees() const { return fovDegrees_; }
    float flySpeed() const { return flySpeed_; }

    void setSnapView(float targetYaw, float targetPitch) {
        yaw_ = targetYaw;
        pitch_ = targetPitch;
    }

private:
    Vec3 target_ = {0.0f, 0.35f, 0.0f};
    float distance_ = 2.5f;
    float yaw_ = 25.0f;
    float pitch_ = 22.0f;
    float fovDegrees_ = 50.0f;
    float flySpeed_ = 2.0f;
};

} // namespace vparticles::viewer
