#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

class Camera;

class Demo {
public:
    virtual ~Demo() = default;

    // Core functions
    virtual const char* getName() const = 0;
    virtual bool is3D() const = 0;  // Is this a 3D demo that uses the camera?
    virtual void update(float deltaTime) = 0;
    virtual void render(uchar4* d_out, int width, int height) = 0;
    virtual void renderUI() = 0;  // ImGui controls for this demo
    virtual void reset() = 0;
    
    // Camera access for 3D demos
    virtual void setCamera(Camera* cam) {}

    // Input event handlers
    virtual void onMouseClick(int button, int state, int x, int y) {}
    virtual void onMouseDrag(int x, int y) {}
    virtual void onMouseWheel(int wheel, int direction, int x, int y) {}
    virtual void onKeyPress(unsigned char key) {}
    virtual void onSpecialKey(int key) {}  // Arrow keys, function keys, etc.
};

