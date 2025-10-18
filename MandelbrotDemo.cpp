#include "MandelbrotDemo.h"
#include <imgui.h>
#include <GL/freeglut.h>
#include <iostream>

// Forward declare CUDA kernel launcher
extern "C" void launchMandelbrotKernel(uchar4* ptr, unsigned int width, unsigned int height, float zoom, float centerX, float centerY);

void MandelbrotDemo::update(float deltaTime) {
    // Mandelbrot is static, no animation needed
}

void MandelbrotDemo::render(uchar4* d_out, int width, int height) {
    windowWidth = width;
    windowHeight = height;
    launchMandelbrotKernel(d_out, width, height, zoom, centerX, centerY);
}

void MandelbrotDemo::renderUI() {
    ImGui::Text("Navigation:");
    ImGui::BulletText("Scroll wheel to zoom");
    ImGui::BulletText("Click and drag to pan");
    ImGui::BulletText("Arrow keys to pan");
    ImGui::Separator();
    
    ImGui::SliderFloat("Zoom", &zoom, 0.1f, 100000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Center X", &centerX, -2.0f, 2.0f, "%.8f");
    ImGui::SliderFloat("Center Y", &centerY, -2.0f, 2.0f, "%.8f");
    
    ImGui::Separator();
    ImGui::Text("Famous Locations:");
    
    if (ImGui::Button("Misiurewicz Point", ImVec2(200, 0))) {
        centerX = -0.77568377f;
        centerY = 0.13646737f;
        zoom = 1.0f;
    }
    
    if (ImGui::Button("Minibrot", ImVec2(200, 0))) {
        centerX = -0.7436439f;
        centerY = 0.1318259f;
        zoom = 1.0f;
    }
    
    if (ImGui::Button("Filaments", ImVec2(200, 0))) {
        centerX = -0.761574f;
        centerY = -0.0847596f;
        zoom = 1.0f;
    }
    
    if (ImGui::Button("Seahorse Valley", ImVec2(200, 0))) {
        centerX = 0.3750001200618655f;
        centerY = -0.2166393884377127f;
        zoom = 1.0f;
    }
    
    ImGui::Separator();
    ImGui::Text("Current View:");
    ImGui::Text("Zoom: %.2f", zoom);
    ImGui::Text("Center: (%.8f, %.8f)", centerX, centerY);
}

void MandelbrotDemo::reset() {
    zoom = 1.0f;
    centerX = -0.77568377f;  // Misiurewicz Point
    centerY = 0.13646737f;
}

void MandelbrotDemo::onMouseClick(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            isDragging = true;
            lastMouseX = x;
            lastMouseY = y;
        } else if (state == GLUT_UP) {
            isDragging = false;
        }
    }
}

void MandelbrotDemo::onMouseDrag(int x, int y) {
    if (isDragging) {
        int dx = x - lastMouseX;
        int dy = y - lastMouseY;
        
        // Convert pixel movement to complex plane movement
        // Scale by zoom and aspect ratio
        float scale = 2.0f / zoom;
        float moveX = -(dx / (float)(windowHeight / 2)) * scale;
        float moveY = -(dy / (float)(windowHeight / 2)) * scale;  // Negative to match intuitive drag
        
        centerX += moveX;
        centerY += moveY;
        
        lastMouseX = x;
        lastMouseY = y;
    }
}

void MandelbrotDemo::onMouseWheel(int wheel, int direction, int x, int y) {
    // Zoom in/out without changing center
    float zoomFactor = (direction > 0) ? 1.2f : 0.8f;
    zoom *= zoomFactor;
    
    // Clamp zoom
    if (zoom < 0.1f) zoom = 0.1f;
    if (zoom > 100000.0f) zoom = 100000.0f;
}

void MandelbrotDemo::onSpecialKey(int key) {
    // Arrow keys for panning
    float moveSpeed = 0.1f / zoom;
    
    switch (key) {
        case GLUT_KEY_UP:
            centerY -= moveSpeed;
            break;
        case GLUT_KEY_DOWN:
            centerY += moveSpeed;
            break;
        case GLUT_KEY_LEFT:
            centerX -= moveSpeed;
            break;
        case GLUT_KEY_RIGHT:
            centerX += moveSpeed;
            break;
    }
}

