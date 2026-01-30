#pragma once

#include <GL/glew.h>
#include "Vec.h"
#include "PointRenderer.h"

class Camera;

// Screen-space fluid surface renderer using curvature flow smoothing
// Based on "Screen Space Fluid Rendering with Curvature Flow" (van der Laan, Green, Sainz 2010)
class SurfaceRenderer {
public:
    SurfaceRenderer();
    ~SurfaceRenderer();

    // Initialize with particle layout (must match PointRenderer layout)
    bool init(int maxParticles, const PointRenderer::AttribLayout& layout);
    
    // Render fluid surface from particle VBO
    void render(Camera* camera, GLuint particleVBO, int particleCount,
                const Vec3& lightDir, int width, int height);
    
    // Fluid rendering parameters
    struct FluidStyle {
        // Curvature flow smoothing
        int smoothingIterations = 50;       // Number of curvature flow iterations
        float smoothingDt = 0.0005f;        // Timestep per iteration
        float smoothingZContrib = 10.0f;    // Adaptive timestep based on depth gradient
        
        // Particle rendering
        float particleScale = 2.0f;         // Scale factor for particle size in depth pass
        float thicknessScale = 3.0f;        // Scale factor for thickness pass particles
        
        // Fluid appearance (Beer's law absorption)
        Vec3 absorptionCoeff = Vec3(0.6f, 0.2f, 0.05f);  // RGB absorption (higher = more absorbed)
        float absorptionScale = 1.0f;       // Overall absorption multiplier
        
        // Lighting
        float specularPower = 64.0f;
        float fresnelPower = 5.0f;
        float ambientStrength = 0.3f;
    };
    
    FluidStyle& getStyle() { return style; }
    const FluidStyle& getStyle() const { return style; }
    
    void cleanup();

private:
    void initShaders();
    void initFramebuffers(int width, int height);
    void setupParticleVAO(GLuint particleVBO);
    
    void renderDepthPass(Camera* camera, int particleCount, int width, int height);
    void renderThicknessPass(Camera* camera, int particleCount, int width, int height);
    void runCurvatureFlowCompute(Camera* camera, int width, int height);
    void renderComposite(Camera* camera, const Vec3& lightDir, int width, int height);

    // Particle VAO (references external VBO)
    GLuint particleVAO = 0;
    GLuint currentVBO = 0;
    int maxParticles = 0;
    PointRenderer::AttribLayout layout;

    // Framebuffers and textures
    GLuint depthFBO = 0;
    GLuint thicknessFBO = 0;
    
    GLuint depthTexture = 0;        // Linear eye-space depth (ping)
    GLuint depthTexture2 = 0;       // Linear eye-space depth (pong)
    GLuint thicknessTexture = 0;    // Accumulated thickness
    GLuint depthRBO = 0;            // Depth renderbuffer for depth testing
    
    int fbWidth = 0;
    int fbHeight = 0;

    // Shaders
    GLuint depthShader = 0;         // Render particles to depth
    GLuint thicknessShader = 0;     // Render particles to thickness (additive)
    GLuint curvatureFlowCompute = 0;// Curvature flow smoothing
    GLuint compositeShader = 0;     // Final shading with normals + absorption

    // Fullscreen quad VAO
    GLuint quadVAO = 0;
    GLuint quadVBO = 0;

    FluidStyle style;
};
