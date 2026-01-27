#pragma once

#include <GL/glew.h>
#include "Vec.h"

class Camera;

// Renders point sprites (balls, particles) with various shaders
class PointRenderer {
public:
    enum class Mode {
        Basic,      // Solid colored spheres
        Textured,   // Textured spheres (no rotation)
        Ball        // Textured with quaternion rotation
    };

    // Vertex attribute layout for the VBO
    struct AttribLayout {
        int strideFloats;    // Total floats per point
        int posOffset;       // Offset to xyz (floats)
        int radiusOffset;    // Offset to radius
        int colorOffset;     // Offset to rgb
        int quatOffset;      // Offset to quaternion (-1 if none)
    };

    PointRenderer();
    ~PointRenderer();

    // Initialize with layout and max point count
    bool init(int maxPoints, const AttribLayout& layout);
    
    // Get VBO handle for CUDA registration
    GLuint getVBO() const { return vbo; }
    
    // Resize VBO (invalidates CUDA registration)
    void resize(int newMaxPoints);
    
    // Render points with specified mode
    void render(Camera* camera, int count, Mode mode, GLuint texture, 
                const Vec3& lightDir, int viewportWidth, int viewportHeight);
    
    // Shadow pass rendering
    void renderShadowPass(const float* lightViewProj, int count, float pointScale);
    
    void cleanup();

private:
    void initShaders();
    void setupVertexAttributes();

    GLuint vao = 0;
    GLuint vbo = 0;
    int maxPoints = 0;
    AttribLayout layout;

    // Separate shaders for each mode
    GLuint shaderBasic = 0;
    GLuint shaderTextured = 0;
    GLuint shaderBall = 0;
    GLuint shaderShadow = 0;
};
