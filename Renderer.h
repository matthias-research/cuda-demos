#pragma once

#include <GL/glew.h>
#include "Mesh.h"
#include "Camera.h"

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    // Initialize shaders
    bool init();
    
    // Render a mesh with the camera
    void renderMesh(const Mesh& mesh, Camera* camera, int width, int height, float scale = 1.0f, const float* modelTransform = nullptr, float rotationX = 0.0f);
    
    // Material properties
    struct Material {
        float ambientStrength = 0.3f;
        float specularStrength = 0.5f;
        float shininess = 32.0f;
    };
    
    void setMaterial(const Material& mat) { material = mat; }
    Material& getMaterial() { return material; }
    
    // Light properties
    struct Light {
        float x = 5.0f;
        float y = 5.0f;
        float z = 5.0f;
    };
    
    void setLight(const Light& l) { light = l; }
    Light& getLight() { return light; }
    
    // Cleanup
    void cleanup();
    
private:
    void setupMatrices(Camera* camera, int width, int height);
    
    GLuint shaderProgram = 0;
    Material material;
    Light light;
    
    // Cached matrices
    float modelMatrix[16];
    float viewMatrix[16];
    float projectionMatrix[16];
};
