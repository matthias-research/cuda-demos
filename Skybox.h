#pragma once

#include <GL/glew.h>
#include "Camera.h"

class Skybox {
public:
    Skybox();
    ~Skybox();
    
    // Load from BMP cubemap cross image - auto-detects face size
    bool loadFromBMP(const char* filepath);
    
    // Render as background (always behind everything)
    void render(Camera* camera);
    
    void cleanup();
    
private:
    GLuint cubemapTexture;
    GLuint vao, vbo;
    GLuint shaderProgram;
    bool initialized;
    
    // Helper to create shaders
    bool createShaders();
    bool createCubeGeometry();
};
