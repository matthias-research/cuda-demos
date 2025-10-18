#pragma once

#include "Demo.h"
#include "Camera.h"
#include <GL/glew.h>

class BoxesDemo : public Demo {
private:
    // Cube geometry
    GLuint vao, vbo;
    
    // Shader program
    GLuint shaderProgram;
    
    // Framebuffer for rendering
    GLuint fbo, renderTexture;
    int fbWidth, fbHeight;
    
    // Animation
    float rotation = 0.0f;
    float rotationSpeed = 45.0f; // degrees per second
    
    // Camera (reference to main camera)
    Camera* camera = nullptr;
    
    // Lighting
    float lightPosX = 5.0f;
    float lightPosY = 5.0f;
    float lightPosZ = 5.0f;
    
    // Material
    float ambientStrength = 0.1f;
    float specularStrength = 0.5f;
    float shininess = 32.0f;
    
    void initCube();
    void initShaders();
    void initFramebuffer(int width, int height);
    void cleanupGL();

public:
    BoxesDemo();
    ~BoxesDemo();
    
    const char* getName() const override { return "3D Boxes (OpenGL)"; }
    bool is3D() const override { return true; }
    void setCamera(Camera* cam) override { camera = cam; }
    void update(float deltaTime) override;
    void render(uchar4* d_out, int width, int height) override;
    void renderUI() override;
    void reset() override;
};
