#pragma once

#include "Demo.h"
#include "Camera.h"
#include "Vec.h"
#include "BVH.h"
#include "Scene.h"
#include "Renderer.h"
#include "Skybox.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>

struct Ball {
    Vec3 pos;       // Position
    Vec3 vel;       // Velocity
    Quat quat;      // Quaternion orientation
    Vec3 angVel;    // Angular velocity
    Vec3 color;     // Color
    float radius;   // Size
};

// CUDA physics functions (implemented in BallsDemo.cu)
void initCudaPhysics(int numBalls, Bounds3 sceneBounds, Bounds3 ballsBounds, float minRadius, float maxRadius, GLuint vbo, cudaGraphicsResource** vboResource, BVHBuilder* bvhBuilder, Scene* scene);
void updateCudaPhysics(float dt, Vec3 gravity, float friction, float terminalVelocity, float bounce, Bounds3 sceneBounds, cudaGraphicsResource* vboResource, bool useBVH);
void cleanupCudaPhysics(cudaGraphicsResource* vboResource);
bool cudaRaycast(const Ray& ray, float& t);

class BallsDemo : public Demo {
private:
    int numBalls = 10000;  // Start with more balls to showcase GPU power
    float gravity = 9.8f;
    float bounce = 0.85f;  // Coefficient of restitution
    float friction = 1.0f; // no friction
    float terminalVelocity = 10.0f;
    
    // Ball size parameters
    float minRadius = 0.25f;
    float maxRadius = 0.25f;
    
    // Camera clipping planes
    float cameraNear = 0.1f;
    float cameraFar = 1000.0f;
    
    // Simulation bounds (walls and floor, open ceiling)
    Bounds3 sceneBounds = Bounds3(Vec3(-600.0f, 0.0f, -310.0f), Vec3(600.0f, 600.0f, 310.0f));
    
    // Initial ball spawn region
//    Bounds3 ballsBounds = Bounds3(Vec3(-100.0f, 100.0f, -200.0f), Vec3(0.0f, 300.0f, 200.0f));
    Bounds3 ballsBounds = Bounds3(Vec3(-200.0f, 250.0f, -100.0f), Vec3(200.0f, 300.0f, 100.0f));

    // OpenGL resources
    GLuint vao, vbo;
    GLuint ballShader;
    GLuint ballShadowShader;
    GLuint shadowFBO, shadowTexture;
    int shadowWidth, shadowHeight;
    
    // CUDA resources
    cudaGraphicsResource* cudaVboResource = nullptr;
    bool useCuda = true;
    bool useBVH = false;  // Toggle between BVH and hash grid collision
    BVHBuilder* bvhBuilder = nullptr;
    
    // Performance tracking
    float lastUpdateTime = 0.0f;
    float fps = 0.0f;
    bool paused = false;
    
    // Camera (reference to main camera)
    Camera* camera = nullptr;
    
    // Lighting

    // To the light (OpenGl convention)
    Vec3 lightDir = Vec3(0.1f, 0.1f, 0.5f).normalized();
    bool useShadows = false;  // Toggle shadow mapping on/off

    // Static scene rendering
    Scene* scene = nullptr;
    Renderer* renderer = nullptr;
    bool showScene = false;
    bool useBakedLighting = false;
    
    // Skybox
    Skybox* skybox = nullptr;
    bool showSkybox = true;
    
    void initBalls();
    void initGL();
    void initShaders();
    void initShadowBuffer(int width, int height);
    void renderShadows(int width, int height);
    void cleanupGL();

public:
    BallsDemo();
    ~BallsDemo();
    
    const char* getName() const override { return "3D Bouncing Balls"; }
    bool is3D() const override { return true; }
    void setCamera(Camera* cam) override { camera = cam; }
    bool raycast(const Vec3& orig, const Vec3& dir, float& t) override;
    void update(float deltaTime) override;
    void render(uchar4* d_out, int width, int height) override;
    void render3D(int width, int height) override;
    void renderUI() override;
    void reset() override;
    void onKeyPress(unsigned char key) override;
};

