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
    
    // Shadow map data
    struct ShadowMap {
        GLuint texture = 0;
        float lightMatrix[16];  // Light view-projection matrix
        float bias = 0.005f;
    };
    
    // Render a mesh with the camera
    void renderMesh(const Mesh& mesh, Camera* camera, int width, int height, float scale = 1.0f, const float* modelTransform = nullptr, float rotationX = 0.0f, const ShadowMap* shadowMap = nullptr);
    
    // Material properties
    struct Material {
        float ambientStrength = 0.2f; 
        float specularStrength = 0.3f;
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
