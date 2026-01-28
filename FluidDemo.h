#pragma once

#include <string>
#include <memory>
#include "Vec.h"
#include "Demo.h"
#include "Camera.h"
#include "Scene.h"
#include "Renderer.h"
#include "PointRenderer.h"
#include "Skybox.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct FluidDemoDescriptor {
    int numParticles = 100000;
    float gravity = 9.8f;
    float particleRadius = 0.2f;
    float kernelRadius = 0.6f;   // SPH kernel support radius (h)
    float maxVelocity = 100.0f;

    std::string meshName = "";
    Bounds3 sceneBounds = Bounds3(Vec3(-50.0f, 0.0f, -50.0f), Vec3(50.0f, 100.0f, 50.0f));
    Bounds3 spawnBounds = Bounds3(Vec3(-10.0f, 20.0f, -10.0f), Vec3(10.0f, 60.0f, 10.0f));

    // Lighting
    float meshAmbient = 0.3f;
    float lightAzimuth = 1.2f;
    float lightElevation = 1.1f;

    // Camera presets
    Vec3 cameraPos = Vec3(0.0f, 30.0f, 80.0f);
    Vec3 cameraLookAt = Vec3(0.0f, 20.0f, 0.0f);
    float cameraNear = 0.1f;
    float cameraFar = 1000.0f;

    void setupDefaultScene() {
        numParticles = 50000;
        gravity = 9.8f;
        particleRadius = 0.3f;
        kernelRadius = 0.9f;  // Typically 2-4x particle radius
        maxVelocity = 50.0f;

        sceneBounds = Bounds3(Vec3(-30.0f, 0.0f, -30.0f), Vec3(30.0f, 80.0f, 30.0f));
        spawnBounds = Bounds3(Vec3(-10.0f, 30.0f, -10.0f), Vec3(10.0f, 70.0f, 10.0f));

        cameraPos = Vec3(0.0f, 40.0f, 100.0f);
        cameraLookAt = Vec3(0.0f, 20.0f, 0.0f);
    }
};


struct FluidDeviceData;
class CudaMeshes;
class CudaHash;


class FluidDemo : public Demo {
public:
    FluidDemo(const FluidDemoDescriptor& desc);
    ~FluidDemo() override;

    const char* getName() const override { return customName.c_str(); }
    void setName(const std::string& name) { customName = name; }
    bool is3D() const override { return true; }
    void setCamera(Camera* cam) override { camera = cam; }
    void applyCameraSettings() { if (camera) camera->lookAt(demoDesc.cameraPos, demoDesc.cameraLookAt); }
    
    void update(float deltaTime) override;
    void render(uchar4* d_out, int width, int height) override;
    void render3D(int width, int height) override;
    void renderUI() override;
    void reset() override;
    
    bool raycast(const Vec3& orig, const Vec3& dir, float& t) override;
    void onKeyPress(unsigned char key) override;

    // CUDA physics functions (implemented in FluidDemo.cu)
    void initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, Scene* scene);
    void updateCudaPhysics(float dt, cudaGraphicsResource* vboResource);
    void cleanupCudaPhysics(cudaGraphicsResource* vboResource);
    bool cudaRaycast(const Ray& ray, float& t);

private:
    void ensureSceneLoaded();
    void initParticles();

    FluidDemoDescriptor demoDesc;
    std::string customName = "Fluid Demo";
    int lastInitializedParticleCount = -1;

    // Point rendering
    PointRenderer* particleRenderer = nullptr;
    
    // CUDA resources
    cudaGraphicsResource* cudaVboResource = nullptr;
    std::shared_ptr<FluidDeviceData> deviceData = nullptr;
    std::shared_ptr<CudaMeshes> meshes = nullptr;
    std::shared_ptr<CudaHash> hash = nullptr;

    // Performance tracking
    float lastUpdateTime = 0.0f;
    float fps = 0.0f;
    bool paused = true;

    // Camera
    Camera* camera = nullptr;

    // Lighting
    Vec3 lightDir = Vec3(0.3f, 0.8f, 0.3f).normalized();

    // Static scene rendering
    Scene* scene = nullptr;
    Renderer* meshRenderer = nullptr;
    bool showScene = false;
    bool sceneLoaded = false;

    // Skybox
    Skybox* skybox = nullptr;
    bool showSkybox = true;
};
