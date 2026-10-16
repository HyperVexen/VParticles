#pragma once

#include <glad/glad.h>
#include <iostream>
#include <string>

namespace vparticles::viewer {

inline const char* kParticleVertexShaderSrc = R"(#version 450 core
layout(location = 0) in vec4 in_Position; // xyz = world pos, w = size
layout(location = 1) in vec4 in_Color;    // rgba

uniform mat4  uViewProj;
uniform vec3  uEyePos;
uniform float uViewportHeight;
uniform float uPointScale;

out vec4 vColor;

void main() {
    vec4 worldPos = vec4(in_Position.xyz, 1.0);
    gl_Position = uViewProj * worldPos;

    float dist = length(uEyePos - in_Position.xyz);
    if (dist < 0.01) dist = 0.01;

    // Attenuate point size by distance to camera
    float pSize = ((in_Position.w * uPointScale) * uViewportHeight) / (dist * 1.2);
    gl_PointSize = clamp(pSize, 1.5, 256.0);

    vColor = in_Color;
}
)";

inline const char* kParticleFragmentShaderSrc = R"(#version 450 core
in vec4 vColor;
out vec4 fragColor;

uniform int uIsGranular;

void main() {
    // gl_PointCoord has range [0, 1] inside point sprite
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float distSq = dot(coord, coord);
    if (distSq > 1.0) {
        discard;
    }

    if (uIsGranular != 0) {
        // Shaded 3D sphere representation for granular grains
        float nz = sqrt(1.0 - distSq);
        vec3 N = vec3(coord.x, -coord.y, nz);
        vec3 L = normalize(vec3(0.5, 0.8, 0.6));
        float diff = max(0.35, dot(N, L));
        float spec = pow(max(0.0, dot(reflect(-L, N), vec3(0.0, 0.0, 1.0))), 16.0) * 0.25;
        fragColor = vec4(vColor.rgb * diff + vec3(spec), 1.0);
    } else {
        // Smooth circular alpha falloff for effects
        float shape = smoothstep(1.0, 0.0, distSq);
        float alpha = vColor.a * shape;
        fragColor = vec4(vColor.rgb, alpha);
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
        uPointScaleLoc_     = glGetUniformLocation(id_, "uPointScale");
        uIsGranularLoc_     = glGetUniformLocation(id_, "uIsGranular");

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

    void setPointScale(float s) const {
        if (uPointScaleLoc_ >= 0) glUniform1f(uPointScaleLoc_, s);
    }

    void setIsGranular(bool isGranular) const {
        if (uIsGranularLoc_ >= 0) glUniform1i(uIsGranularLoc_, isGranular ? 1 : 0);
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
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            std::cerr << "Shader compile error (" << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << "):\n"
                      << log << std::endl;
            glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    GLuint id_ = 0;
    GLint uViewProjLoc_ = -1;
    GLint uEyePosLoc_ = -1;
    GLint uViewportHeightLoc_ = -1;
    GLint uPointScaleLoc_ = -1;
    GLint uIsGranularLoc_ = -1;
};

} // namespace vparticles::viewer
