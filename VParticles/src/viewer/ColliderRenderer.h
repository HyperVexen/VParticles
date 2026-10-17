#pragma once

#include <glad/glad.h>
#include "Math3D.h"
#include <cmath>
#include <vector>
#include <iostream>

namespace vparticles::viewer {

class ColliderRenderer {
public:
    ColliderRenderer() = default;
    ~ColliderRenderer() {
        if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
        if (vbo_ != 0) glDeleteBuffers(1, &vbo_);
        if (program_ != 0) glDeleteProgram(program_);
    }

    bool init() {
        const char* vsSrc = R"(#version 450 core
layout(location = 0) in vec3 in_Position;
uniform mat4 uViewProj;
void main() {
    gl_Position = uViewProj * vec4(in_Position, 1.0);
}
)";

        const char* fsSrc = R"(#version 450 core
uniform vec4 uColor;
out vec4 fragColor;
void main() {
    fragColor = uColor;
}
)";

        GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
        if (vs == 0 || fs == 0) return false;

        program_ = glCreateProgram();
        glAttachShader(program_, vs);
        glAttachShader(program_, fs);
        glLinkProgram(program_);

        glDeleteShader(vs);
        glDeleteShader(fs);

        uViewProjLoc_ = glGetUniformLocation(program_, "uViewProj");
        uColorLoc_    = glGetUniformLocation(program_, "uColor");

        glGenVertexArrays(1, &vao_);
        glBindVertexArray(vao_);

        glGenBuffers(1, &vbo_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * 2048, nullptr, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, reinterpret_cast<void*>(0));

        glBindVertexArray(0);
        return true;
    }

    void drawSphere(const Mat4& vp, const Float3& center, float radius, const float color[4]) {
        if (program_ == 0 || radius <= 0.001f) return;

        std::vector<float> lines;
        const int segments = 40;
        const float dTheta = 2.0f * kPi / static_cast<float>(segments);

        // 1. Equatorial ring (XZ plane)
        for (int i = 0; i < segments; ++i) {
            float t0 = i * dTheta;
            float t1 = (i + 1) * dTheta;
            lines.push_back(center.x + radius * std::cos(t0));
            lines.push_back(center.y);
            lines.push_back(center.z + radius * std::sin(t0));

            lines.push_back(center.x + radius * std::cos(t1));
            lines.push_back(center.y);
            lines.push_back(center.z + radius * std::sin(t1));
        }

        // 2. Meridian ring 1 (XY plane)
        for (int i = 0; i < segments; ++i) {
            float t0 = i * dTheta;
            float t1 = (i + 1) * dTheta;
            lines.push_back(center.x + radius * std::cos(t0));
            lines.push_back(center.y + radius * std::sin(t0));
            lines.push_back(center.z);

            lines.push_back(center.x + radius * std::cos(t1));
            lines.push_back(center.y + radius * std::sin(t1));
            lines.push_back(center.z);
        }

        // 3. Meridian ring 2 (YZ plane)
        for (int i = 0; i < segments; ++i) {
            float t0 = i * dTheta;
            float t1 = (i + 1) * dTheta;
            lines.push_back(center.x);
            lines.push_back(center.y + radius * std::sin(t0));
            lines.push_back(center.z + radius * std::cos(t0));

            lines.push_back(center.x);
            lines.push_back(center.y + radius * std::sin(t1));
            lines.push_back(center.z + radius * std::cos(t1));
        }

        // 4. Latitude rings at +45 deg and -45 deg
        float rLat = radius * 0.7071f;
        float yLat = radius * 0.7071f;
        for (int sign = -1; sign <= 1; sign += 2) {
            float curY = center.y + sign * yLat;
            for (int i = 0; i < segments; ++i) {
                float t0 = i * dTheta;
                float t1 = (i + 1) * dTheta;
                lines.push_back(center.x + rLat * std::cos(t0));
                lines.push_back(curY);
                lines.push_back(center.z + rLat * std::sin(t0));

                lines.push_back(center.x + rLat * std::cos(t1));
                lines.push_back(curY);
                lines.push_back(center.z + rLat * std::sin(t1));
            }
        }

        uploadAndDraw(vp, lines, color, 2.0f);
    }

    void drawBox(const Mat4& vp, const Float3& minB, const Float3& maxB, const float color[4]) {
        if (program_ == 0) return;

        std::vector<float> lines;
        // 8 vertices of box
        Float3 c[8] = {
            {minB.x, minB.y, minB.z},
            {maxB.x, minB.y, minB.z},
            {maxB.x, maxB.y, minB.z},
            {minB.x, maxB.y, minB.z},
            {minB.x, minB.y, maxB.z},
            {maxB.x, minB.y, maxB.z},
            {maxB.x, maxB.y, maxB.z},
            {minB.x, maxB.y, maxB.z}
        };

        // 12 edges
        int edges[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom face
            {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top face
            {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Pillars
        };

        for (int e = 0; e < 12; ++e) {
            lines.push_back(c[edges[e][0]].x);
            lines.push_back(c[edges[e][0]].y);
            lines.push_back(c[edges[e][0]].z);

            lines.push_back(c[edges[e][1]].x);
            lines.push_back(c[edges[e][1]].y);
            lines.push_back(c[edges[e][1]].z);
        }

        uploadAndDraw(vp, lines, color, 1.5f);
    }

    void drawCircle(const Mat4& vp, const Float3& center, float radius, const float color[4]) {
        if (program_ == 0 || radius <= 0.001f) return;

        std::vector<float> lines;
        const int segments = 48;
        const float dTheta = 2.0f * kPi / static_cast<float>(segments);

        for (int i = 0; i < segments; ++i) {
            float t0 = i * dTheta;
            float t1 = (i + 1) * dTheta;
            lines.push_back(center.x + radius * std::cos(t0));
            lines.push_back(center.y);
            lines.push_back(center.z + radius * std::sin(t0));

            lines.push_back(center.x + radius * std::cos(t1));
            lines.push_back(center.y);
            lines.push_back(center.z + radius * std::sin(t1));
        }

        uploadAndDraw(vp, lines, color, 2.0f);
    }

private:
    void uploadAndDraw(const Mat4& vp, const std::vector<float>& lines, const float color[4], float lineWidth) {
        if (lines.empty()) return;

        glUseProgram(program_);
        glUniformMatrix4fv(uViewProjLoc_, 1, GL_FALSE, vp.data());
        glUniform4fv(uColorLoc_, 1, color);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, lines.size() * sizeof(float), lines.data());

        glLineWidth(lineWidth);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(lines.size() / 3));

        glBindVertexArray(0);
        glUseProgram(0);
    }

    GLuint compileShader(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);

        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[1024];
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            std::cerr << "ColliderRenderer shader error: " << log << std::endl;
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    GLuint program_ = 0;
    GLuint vao_ = 0;
    GLuint vbo_ = 0;
    GLint uViewProjLoc_ = -1;
    GLint uColorLoc_ = -1;
};

} // namespace vparticles::viewer
