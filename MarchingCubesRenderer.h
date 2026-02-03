#pragma once

#include <GL/glew.h>
#include "Vec.h"

class Camera;
class MarchingCubesSurface;

// Renders marching cubes surface mesh from GPU-generated VBOs
class MarchingCubesRenderer {
public:
    MarchingCubesRenderer();
    ~MarchingCubesRenderer();

    bool init();

    // Render from MarchingCubesSurface VBOs
    void render(Camera* camera, MarchingCubesSurface* surface,
                const Vec3& lightDir, int width, int height);

    // Material properties
    struct Material {
        Vec3 baseColor = Vec3(0.3f, 0.5f, 0.9f);  // Blue-ish fluid color
        float ambientStrength = 0.2f;
        float specularStrength = 0.6f;
        float shininess = 64.0f;
        float fresnelPower = 3.0f;
        float alpha = 0.85f;
    };

    Material& getMaterial() { return material; }
    const Material& getMaterial() const { return material; }

    void cleanup();

private:
    void setupVAO(GLuint verticesVbo, GLuint normalsVbo);

    GLuint shaderProgram = 0;
    GLuint vao = 0;
    GLuint currentVerticesVbo = 0;
    GLuint currentNormalsVbo = 0;

    Material material;
};
