#pragma once

#include "Demo.h"
#include "Camera.h"
#include "Vec.h"
#include "BVH.h"
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
extern "C" void initCudaPhysics(int numBalls, float roomSize, GLuint vbo, cudaGraphicsResource** vboResource, BVHBuilder* bvhBuilder);
extern "C" void updateCudaPhysics(float dt, Vec3 gravity, float friction, float bounce, float roomSize, cudaGraphicsResource* vboResource, bool useBVH);
extern "C" void cleanupCudaPhysics(cudaGraphicsResource* vboResource);

class BallsDemo : public Demo {
private:
    int numBalls = 1000;  // Start with more balls to showcase GPU power
    float gravity = 9.8f;
    float bounce = 0.85f;  // Coefficient of restitution
    float friction = 0.99f;
    
    // Simulation bounds
    float roomSize = 20.0f;  // Double size room for more balls
    
    // OpenGL resources
    GLuint vao, vbo;
    GLuint sphereShader;
    GLuint fbo, renderTexture;
    int fbWidth, fbHeight;
    
    // CUDA resources
    cudaGraphicsResource* cudaVboResource = nullptr;
    bool useCuda = true;
    bool useBVH = false;  // Toggle between BVH and hash grid collision
    BVHBuilder* bvhBuilder = nullptr;
    
    // Performance tracking
    float lastUpdateTime = 0.0f;
    float fps = 0.0f;
    
    // Camera (reference to main camera)
    Camera* camera = nullptr;
    
    // Lighting
    float lightPosX = 5.0f;
    float lightPosY = 8.0f;
    float lightPosZ = 5.0f;
    
    void initBalls();
    void initGL();
    void initShaders();
    void initFramebuffer(int width, int height);
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
    void renderUI() override;
    void reset() override;
};

