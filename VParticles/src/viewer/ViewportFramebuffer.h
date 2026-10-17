#pragma once

#include <glad/glad.h>
#include <iostream>

namespace vparticles::viewer {

class ViewportFramebuffer {
public:
    ViewportFramebuffer() = default;
    ~ViewportFramebuffer() {
        destroy();
    }

    bool init(int width, int height) {
        width_ = width;
        height_ = height;

        glGenFramebuffers(1, &fbo_);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

        // 1. Color attachment texture
        glGenTextures(1, &colorTexture_);
        glBindTexture(GL_TEXTURE_2D, colorTexture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture_, 0);

        // 2. Depth and Stencil renderbuffer
        glGenRenderbuffers(1, &rboDepth_);
        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width_, height_);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rboDepth_);

        // 3. Completeness check
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Viewport Framebuffer is incomplete!" << std::endl;
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return false;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return true;
    }

    void resize(int width, int height) {
        if (width <= 0 || height <= 0) return;
        if (width == width_ && height == height_) return;

        width_ = width;
        height_ = height;

        glBindTexture(GL_TEXTURE_2D, colorTexture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBindRenderbuffer(GL_RENDERBUFFER, rboDepth_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width_, height_);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
    }

    void bind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
        glViewport(0, 0, width_, height_);
    }

    void unbind() const {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void destroy() {
        if (colorTexture_ != 0) { glDeleteTextures(1, &colorTexture_); colorTexture_ = 0; }
        if (rboDepth_ != 0)     { glDeleteRenderbuffers(1, &rboDepth_); rboDepth_ = 0; }
        if (fbo_ != 0)          { glDeleteFramebuffers(1, &fbo_); fbo_ = 0; }
        width_ = 0;
        height_ = 0;
    }

    GLuint fbo() const { return fbo_; }
    GLuint texture() const { return colorTexture_; }
    int width() const { return width_; }
    int height() const { return height_; }
    float aspect() const { return height_ > 0 ? (static_cast<float>(width_) / static_cast<float>(height_)) : 1.0f; }

private:
    GLuint fbo_ = 0;
    GLuint colorTexture_ = 0;
    GLuint rboDepth_ = 0;
    int width_ = 0;
    int height_ = 0;
};

} // namespace vparticles::viewer
