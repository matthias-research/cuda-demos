#pragma once

#include "Demo.h"

class MandelbrotDemo : public Demo {
private:
    float zoom = 1.0f;
    float centerX = -0.77568377f;  // Misiurewicz Point
    float centerY = 0.13646737f;
    
    // Mouse interaction state
    int lastMouseX = 0;
    int lastMouseY = 0;
    bool isDragging = false;
    int windowWidth = 1024;
    int windowHeight = 768;

public:
    const char* getName() const override { return "Mandelbrot Fractal"; }
    bool is3D() const override { return false; }
    void update(float deltaTime) override;
    void render(uchar4* d_out, int width, int height) override;
    void renderUI() override;
    void reset() override;
    
    // Input handlers
    void onMouseClick(int button, int state, int x, int y) override;
    void onMouseDrag(int x, int y) override;
    void onMouseWheel(int wheel, int direction, int x, int y) override;
    void onSpecialKey(int key) override;
};

