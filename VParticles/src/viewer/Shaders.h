#pragma once

#include <glad/glad.h>
#include <iostream>
#include <string>

namespace vparticles::viewer {

inline const char* kParticleVertexShaderSrc = R"(#version 450 core
layout(location = 0) in vec4 in_Position; // xyz = world pos, w = physical radius (m)
layout(location = 1) in vec4 in_Color;    // rgba

uniform mat4  uViewProj;
uniform vec3  uEyePos;
uniform float uViewportHeight;
uniform float uDisplayScale;
uniform float uMinPointSize;
uniform float uMaxPointSize;
uniform float uFovRad;
uniform int   uIsOrthographic;
uniform float uOrthoHeight;

out vec4 vColor;

void main() {
    vec4 worldPos = vec4(in_Position.xyz, 1.0);
    gl_Position = uViewProj * worldPos;

    float rWorld = in_Position.w;
    if (rWorld <= 0.0) rWorld = 0.01;
    float diameter = rWorld * 2.0 * uDisplayScale;

    float pixelSize;
    if (uIsOrthographic != 0) {
        float oHeight = max(uOrthoHeight, 0.001);
        pixelSize = (diameter / oHeight) * uViewportHeight;
    } else {
        float dist = length(uEyePos - in_Position.xyz);
        if (dist < 0.01) dist = 0.01;
        float focalLength = uViewportHeight / (2.0 * tan(max(uFovRad, 0.01) * 0.5));
        pixelSize = (diameter / dist) * focalLength;
    }

    gl_PointSize = clamp(pixelSize, uMinPointSize, uMaxPointSize);
    vColor = in_Color;
}
)";

inline const char* kParticleFragmentShaderSrc = R"(#version 450 core
in vec4 vColor;
out vec4 fragColor;

uniform int uRenderStyle; // 0: Shaded Sphere, 1: Flat Disc, 2: Wire Ring

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float distSq = dot(coord, coord);
    if (distSq > 1.0) {
        discard;
    }

    if (uRenderStyle == 0) {
        // Shaded 3D sphere: analytical normal reconstruction
        float nz = sqrt(1.0 - distSq);
        vec3 N = vec3(coord.x, -coord.y, nz);
        vec3 L = normalize(vec3(0.5, 0.8, 0.6));
        float diff = max(0.35, dot(N, L));
        float spec = pow(max(0.0, dot(reflect(-L, N), vec3(0.0, 0.0, 1.0))), 16.0) * 0.35;
        float rim = smoothstep(1.0, 0.85, distSq);
        vec3 finalCol = (vColor.rgb * diff + vec3(spec)) * (0.8 + 0.2 * rim);
        fragColor = vec4(finalCol, vColor.a);
    } else if (uRenderStyle == 1) {
        // Flat circular disc with smooth anti-aliased edge
        float edge = smoothstep(1.0, 0.90, distSq);
        fragColor = vec4(vColor.rgb, vColor.a * edge);
    } else {
        // Wire / diagnostic ring
        float ring = smoothstep(0.65, 0.75, distSq) * smoothstep(1.0, 0.92, distSq);
        if (ring < 0.05) discard;
        fragColor = vec4(vColor.rgb, vColor.a * ring);
    }
}
)";

class ShaderProgram {
public:
    ShaderProgram() = default;
    ~ShaderProgram() {
        if (id_ != 0) {
            glDeleteProgram(id_);
        }
    }

    bool compile(const char* vsSrc, const char* fsSrc) {
        GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
        if (vs == 0) return false;

        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
        if (fs == 0) {
            glDeleteShader(vs);
            return false;
        }

        id_ = glCreateProgram();
        glAttachShader(id_, vs);
        glAttachShader(id_, fs);
        glLinkProgram(id_);

        GLint success = 0;
        glGetProgramiv(id_, GL_LINK_STATUS, &success);
        if (!success) {
            char log[1024];
            glGetProgramInfoLog(id_, sizeof(log), nullptr, log);
            std::cerr << "Shader program link error:\n" << log << std::endl;
            glDeleteProgram(id_);
            id_ = 0;
            glDeleteShader(vs);
            glDeleteShader(fs);
            return false;
        }

        glDeleteShader(vs);
        glDeleteShader(fs);

        uViewProjLoc_       = glGetUniformLocation(id_, "uViewProj");
        uEyePosLoc_         = glGetUniformLocation(id_, "uEyePos");
        uViewportHeightLoc_ = glGetUniformLocation(id_, "uViewportHeight");
        uDisplayScaleLoc_   = glGetUniformLocation(id_, "uDisplayScale");
        uMinPointSizeLoc_   = glGetUniformLocation(id_, "uMinPointSize");
        uMaxPointSizeLoc_   = glGetUniformLocation(id_, "uMaxPointSize");
        uFovRadLoc_         = glGetUniformLocation(id_, "uFovRad");
        uIsOrthographicLoc_ = glGetUniformLocation(id_, "uIsOrthographic");
        uOrthoHeightLoc_    = glGetUniformLocation(id_, "uOrthoHeight");
        uRenderStyleLoc_    = glGetUniformLocation(id_, "uRenderStyle");

        return true;
    }

    void use() const {
        glUseProgram(id_);
    }

    void setViewProj(const float* mat) const {
        if (uViewProjLoc_ >= 0) glUniformMatrix4fv(uViewProjLoc_, 1, GL_FALSE, mat);
    }

    void setEyePos(float x, float y, float z) const {
        if (uEyePosLoc_ >= 0) glUniform3f(uEyePosLoc_, x, y, z);
    }

    void setViewportHeight(float h) const {
        if (uViewportHeightLoc_ >= 0) glUniform1f(uViewportHeightLoc_, h);
    }

    void setDisplayScale(float s) const {
        if (uDisplayScaleLoc_ >= 0) glUniform1f(uDisplayScaleLoc_, s);
    }

    void setMinMaxPointSize(float minSz, float maxSz) const {
        if (uMinPointSizeLoc_ >= 0) glUniform1f(uMinPointSizeLoc_, minSz);
        if (uMaxPointSizeLoc_ >= 0) glUniform1f(uMaxPointSizeLoc_, maxSz);
    }

    void setCameraProjection(float fovRad, bool isOrtho, float orthoHeight) const {
        if (uFovRadLoc_ >= 0) glUniform1f(uFovRadLoc_, fovRad);
        if (uIsOrthographicLoc_ >= 0) glUniform1i(uIsOrthographicLoc_, isOrtho ? 1 : 0);
        if (uOrthoHeightLoc_ >= 0) glUniform1f(uOrthoHeightLoc_, orthoHeight);
    }

    void setRenderStyle(int style) const {
        if (uRenderStyleLoc_ >= 0) glUniform1i(uRenderStyleLoc_, style);
    }

    GLuint id() const { return id_; }

private:
    GLuint compileShader(GLenum type, const char* src) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, nullptr);
        glCompileShader(shader);

        GLint success = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[1024];
            glGetProgramInfoLog(shader, sizeof(log), nullptr, log);
            std::cerr << "Shader compile error (" << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << "):\n"
                      << log << std::endl;
            glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    GLuint id_ = 0;
    GLint uViewProjLoc_       = -1;
    GLint uEyePosLoc_         = -1;
    GLint uViewportHeightLoc_ = -1;
    GLint uDisplayScaleLoc_   = -1;
    GLint uMinPointSizeLoc_   = -1;
    GLint uMaxPointSizeLoc_   = -1;
    GLint uFovRadLoc_         = -1;
    GLint uIsOrthographicLoc_ = -1;
    GLint uOrthoHeightLoc_    = -1;
    GLint uRenderStyleLoc_    = -1;
};

} // namespace vparticles::viewer
