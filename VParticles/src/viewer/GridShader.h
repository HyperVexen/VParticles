#pragma once

#include <glad/glad.h>
#include "Math3D.h"
#include <iostream>

namespace vparticles::viewer {

inline const char* kInfiniteGridVertexShaderSrc = R"(#version 450 core
layout(location = 0) in vec3 in_Pos;

uniform mat4 uInvView;
uniform mat4 uInvProj;

out vec3 vNearPoint;
out vec3 vFarPoint;

vec3 unprojectPoint(float x, float y, float z, mat4 invView, mat4 invProj) {
    vec4 clipPoint = vec4(x, y, z, 1.0);
    vec4 viewPoint = invProj * clipPoint;
    viewPoint /= viewPoint.w;
    vec4 worldPoint = invView * viewPoint;
    return worldPoint.xyz;
}

void main() {
    vec3 p = in_Pos;
    vNearPoint = unprojectPoint(p.x, p.y, 0.0, uInvView, uInvProj);
    vFarPoint  = unprojectPoint(p.x, p.y, 1.0, uInvView, uInvProj);
    gl_Position = vec4(p, 1.0);
}
)";

inline const char* kInfiniteGridFragmentShaderSrc = R"(#version 450 core
in vec3 vNearPoint;
in vec3 vFarPoint;

out vec4 fragColor;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec3 uCameraPos;

vec4 grid(vec3 fragPos3D, float scale, bool drawAxis) {
    vec2 coord = fragPos3D.xz * scale;
    vec2 derivative = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    float line = min(grid.x, grid.y);
    float minimumz = min(derivative.y, 1.0);
    float minimumx = min(derivative.x, 1.0);
    vec4 color = vec4(0.22, 0.23, 0.27, 1.0 - min(line, 1.0));

    // Axis coloring: X axis is Red (z ~ 0), Z axis is Blue (x ~ 0)
    if (drawAxis) {
        if (fragPos3D.z > -0.05 * minimumz && fragPos3D.z < 0.05 * minimumz)
            color = vec4(0.85, 0.22, 0.22, 1.0); // Red X-axis
        if (fragPos3D.x > -0.05 * minimumx && fragPos3D.x < 0.05 * minimumx)
            color = vec4(0.22, 0.55, 0.88, 1.0); // Blue Z-axis
    }
    return color;
}

float computeDepth(vec3 pos, mat4 view, mat4 proj) {
    vec4 clip = proj * view * vec4(pos, 1.0);
    return (clip.z / clip.w) * 0.5 + 0.5;
}

void main() {
    float t = -vNearPoint.y / (vFarPoint.y - vNearPoint.y);
    if (t <= 0.0) {
        discard;
    }

    vec3 fragPos3D = vNearPoint + t * (vFarPoint - vNearPoint);
    gl_FragDepth = computeDepth(fragPos3D, uView, uProj);

    float linearDepth = length(fragPos3D.xz - uCameraPos.xz);
    float fading = max(0.0, 1.0 - (linearDepth / 60.0)); // Radial fog fade at 60m

    // Dual-frequency grid: 1.0m major cells and 0.1m minor cells
    vec4 subGrid  = grid(fragPos3D, 10.0, false);
    vec4 mainGrid = grid(fragPos3D, 1.0, true);

    float alpha = max(subGrid.a * 0.25, mainGrid.a);
    vec3 col = (mainGrid.a > 0.01) ? mainGrid.rgb : subGrid.rgb;

    fragColor = vec4(col, alpha * fading);
    if (fragColor.a < 0.01) {
        discard;
    }
}
)";

class InfiniteGridRenderer {
public:
    InfiniteGridRenderer() = default;
    ~InfiniteGridRenderer() {
        cleanup();
    }

    bool init() {
        cleanup();

        GLuint vs = compileShader(GL_VERTEX_SHADER, kInfiniteGridVertexShaderSrc);
        if (vs == 0) return false;

        GLuint fs = compileShader(GL_FRAGMENT_SHADER, kInfiniteGridFragmentShaderSrc);
        if (fs == 0) {
            glDeleteShader(vs);
            return false;
        }

        program_ = glCreateProgram();
        glAttachShader(program_, vs);
        glAttachShader(program_, fs);
        glLinkProgram(program_);

        glDeleteShader(vs);
        glDeleteShader(fs);

        GLint success = 0;
        glGetProgramiv(program_, GL_LINK_STATUS, &success);
        if (!success) {
            char log[1024];
            glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
            std::cerr << "Grid shader link error:\n" << log << std::endl;
            glDeleteProgram(program_);
            program_ = 0;
            return false;
        }

        locInvView_   = glGetUniformLocation(program_, "uInvView");
        locInvProj_   = glGetUniformLocation(program_, "uInvProj");
        locView_      = glGetUniformLocation(program_, "uView");
        locProj_      = glGetUniformLocation(program_, "uProj");
        locCameraPos_ = glGetUniformLocation(program_, "uCameraPos");

        // Full screen quad in NDC
        const float quadVertices[] = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f
        };

        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        return true;
    }

    void draw(const Mat4& view, const Mat4& proj, const Vec3& cameraPos) {
        if (program_ == 0 || vao_ == 0) return;

        Mat4 invView = invertMatrix(view);
        Mat4 invProj = invertMatrix(proj);

        glUseProgram(program_);

        if (locView_ >= 0)      glUniformMatrix4fv(locView_, 1, GL_FALSE, view.data());
        if (locProj_ >= 0)      glUniformMatrix4fv(locProj_, 1, GL_FALSE, proj.data());
        if (locInvView_ >= 0)   glUniformMatrix4fv(locInvView_, 1, GL_FALSE, invView.data());
        if (locInvProj_ >= 0)   glUniformMatrix4fv(locInvProj_, 1, GL_FALSE, invProj.data());
        if (locCameraPos_ >= 0) glUniform3f(locCameraPos_, cameraPos.x, cameraPos.y, cameraPos.z);

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE); // Don't write to depth buffer so particles can draw behind/in front
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glUseProgram(0);
    }

    void cleanup() {
        if (vao_ != 0) { glDeleteVertexArrays(1, &vao_); vao_ = 0; }
        if (vbo_ != 0) { glDeleteBuffers(1, &vbo_); vbo_ = 0; }
        if (program_ != 0) { glDeleteProgram(program_); program_ = 0; }
    }

private:
    static Mat4 invertMatrix(const Mat4& mat) {
        // Compute 4x4 matrix inverse
        const float* m = mat.data();
        Mat4 inv;
        float* o = inv.data();

        o[0] = m[5]  * m[10] * m[15] - 
               m[5]  * m[11] * m[14] - 
               m[9]  * m[6]  * m[15] + 
               m[9]  * m[7]  * m[14] +
               m[13] * m[6]  * m[11] - 
               m[13] * m[7]  * m[10];

        o[4] = -m[4]  * m[10] * m[15] + 
                m[4]  * m[11] * m[14] + 
                m[8]  * m[6]  * m[15] - 
                m[8]  * m[7]  * m[14] - 
                m[12] * m[6]  * m[11] + 
                m[12] * m[7]  * m[10];

        o[8] = m[4]  * m[9] * m[15] - 
               m[4]  * m[11] * m[13] - 
               m[8]  * m[5] * m[15] + 
               m[8]  * m[7] * m[13] + 
               m[12] * m[5] * m[11] - 
               m[12] * m[7] * m[9];

        o[12] = -m[4]  * m[9] * m[14] + 
                 m[4]  * m[10] * m[13] +
                 m[8]  * m[5] * m[14] - 
                 m[8]  * m[6] * m[13] - 
                 m[12] * m[5] * m[10] + 
                 m[12] * m[6] * m[9];

        o[1] = -m[1]  * m[10] * m[15] + 
                m[1]  * m[11] * m[14] + 
                m[9]  * m[2] * m[15] - 
                m[9]  * m[3] * m[14] - 
                m[13] * m[2] * m[11] + 
                m[13] * m[3] * m[10];

        o[5] = m[0]  * m[10] * m[15] - 
               m[0]  * m[11] * m[14] - 
               m[8]  * m[2] * m[15] + 
               m[8]  * m[3] * m[14] + 
               m[12] * m[2] * m[11] - 
               m[12] * m[3] * m[10];

        o[9] = -m[0]  * m[9] * m[15] + 
                m[0]  * m[11] * m[13] + 
                m[8]  * m[1] * m[15] - 
                m[8]  * m[3] * m[13] - 
                m[12] * m[1] * m[11] + 
                m[12] * m[3] * m[9];

        o[13] = m[0]  * m[9] * m[14] - 
                m[0]  * m[10] * m[13] - 
                m[8]  * m[1] * m[14] + 
                m[8]  * m[2] * m[13] + 
                m[12] * m[1] * m[10] - 
                m[12] * m[2] * m[9];

        o[2] = m[1]  * m[6] * m[15] - 
               m[1]  * m[7] * m[14] - 
               m[5]  * m[2] * m[15] + 
               m[5]  * m[3] * m[14] + 
               m[13] * m[2] * m[7] - 
               m[13] * m[3] * m[6];

        o[6] = -m[0]  * m[6] * m[15] + 
                m[0]  * m[7] * m[14] + 
                m[4]  * m[2] * m[15] - 
                m[4]  * m[3] * m[14] - 
                m[12] * m[2] * m[7] + 
                m[12] * m[3] * m[6];

        o[10] = m[0]  * m[5] * m[15] - 
                m[0]  * m[7] * m[13] - 
                m[4]  * m[1] * m[15] + 
                m[4]  * m[3] * m[13] + 
                m[12] * m[1] * m[7] - 
                m[12] * m[3] * m[5];

        o[14] = -m[0]  * m[5] * m[14] + 
                 m[0]  * m[6] * m[13] + 
                 m[4]  * m[1] * m[14] - 
                 m[4]  * m[2] * m[13] - 
                 m[12] * m[1] * m[6] + 
                 m[12] * m[2] * m[5];

        o[3] = -m[1] * m[6] * m[11] + 
                m[1] * m[7] * m[10] + 
                m[5] * m[2] * m[11] - 
                m[5] * m[3] * m[10] - 
                m[9] * m[2] * m[7] + 
                m[9] * m[3] * m[6];

        o[7] = m[0] * m[6] * m[11] - 
               m[0] * m[7] * m[10] - 
               m[4] * m[2] * m[11] + 
               m[4] * m[3] * m[10] + 
               m[8] * m[2] * m[7] - 
               m[8] * m[3] * m[6];

        o[11] = -m[0] * m[5] * m[11] + 
                 m[0] * m[7] * m[9] + 
                 m[4] * m[1] * m[11] - 
                 m[4] * m[3] * m[9] - 
                 m[8] * m[1] * m[7] + 
                 m[8] * m[3] * m[5];

        o[15] = m[0] * m[5] * m[10] - 
                m[0] * m[6] * m[9] - 
                m[4] * m[1] * m[10] + 
                m[4] * m[2] * m[9] + 
                m[8] * m[1] * m[6] - 
                m[8] * m[2] * m[5];

        float det = m[0] * o[0] + m[1] * o[4] + m[2] * o[8] + m[3] * o[12];
        if (std::abs(det) < 1e-8f) {
            return Mat4::identity();
        }

        float invDet = 1.0f / det;
        for (int i = 0; i < 16; ++i) {
            o[i] *= invDet;
        }

        return inv;
    }

    GLuint compileShader(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);

        GLint success = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[1024];
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            std::cerr << "Grid shader compile error:\n" << log << std::endl;
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;

    GLint locInvView_ = -1;
    GLint locInvProj_ = -1;
    GLint locView_ = -1;
    GLint locProj_ = -1;
    GLint locCameraPos_ = -1;
};

} // namespace vparticles::viewer
