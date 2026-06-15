#pragma once

// Per-instance data written by the GPU kernel into the OpenGL VBO.
// Layout must match the vertex shader attribute locations exactly.
struct alignas(16) InstanceData
{
    float x;    // world position x (pixels)
    float y;    // world position y (pixels)
    float size; // half-size in pixels
    uint8_t r;  // color r [0,255]
    uint8_t g;  // color g [0,255]
    uint8_t b;  // color b [0,255]
    uint8_t a;  // color a [0,255]
};
