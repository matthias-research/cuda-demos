#pragma once

#include <string>
#include <memory>
#include "Vec.h"
#include "Demo.h"
#include "Camera.h"
#include "Scene.h"
#include "Renderer.h"
#include "PointRenderer.h"
#include "SurfaceRenderer.h"
#include "MarchingCubesRenderer.h"
#include "MarchingCubesSurface.h"
#include "Skybox.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Fluid rendering mode
enum class FluidRenderMode {
    Particles,          // Point sprites
    ScreenSpaceSurface, // Screen-space curvature flow
    MarchingCubes       // Marching cubes mesh
};

struct FluidDemoDescriptor 
{
    // Physics
    int numParticles = 100000;
    int numMaxParticles = 1000000;
    float gravity = 9.8f;
    float particleRadius = 0.2f;
    float kernelRadius = 0.6f;   // SPH kernel support radius (h)
    float maxVelocity = 100.0f;

    OBB sourceBounds = OBB(Transform(Identity), Vec3(10.0f, 20.0f, 10.0f));
    float sourceSpeed = 10.0f;
    float sourceDensity = 1.0f;
    OBB sinkBounds = OBB(Transform(Identity), Vec3(Zero));

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

    void setupDebugScene() {
        numParticles = 1;
        gravity = 0.0f;  // No gravity for debugging
        particleRadius = 5.0f;  // Large particle
        kernelRadius = 15.0f;   // Large kernel radius
        maxVelocity = 0.0f;

        sceneBounds = Bounds3(Vec3(-100.0f, -100.0f, -100.0f), Vec3(100.0f, 100.0f, 100.0f));
        spawnBounds = Bounds3(Vec3(0.0f, 30.0f, 0.0f), Vec3(0.0f, 30.0f, 0.0f));  // Single point

        cameraPos = Vec3(0.0f, 30.0f, 60.0f);
        cameraLookAt = Vec3(0.0f, 30.0f, 0.0f);
    }

    void setupTunnelScene() {
        // Start with no particles - they'll be created by the source
        numParticles = 0;
        numMaxParticles = 500000;  // Max capacity for particles
        
        // Physics settings
        gravity = 0.0f;  // No gravity in wind tunnel
        particleRadius = 0.2f;
        kernelRadius = 0.6f;
        maxVelocity = 100.0f;
        
        // Tunnel dimensions: long box along X-axis (smaller for fewer particles)
        float tunnelLength = 50.0f;  // Total length
        float tunnelHeight = 5.0f;   // Y dimension
        float tunnelWidth = 5.0f;    // Z dimension
        
        // Scene bounds: tunnel extends from -tunnelLength/2 to +tunnelLength/2 along X
        sceneBounds = Bounds3(
            Vec3(-tunnelLength / 2.0f, -tunnelHeight / 2.0f, -tunnelWidth / 2.0f),
            Vec3(tunnelLength / 2.0f, tunnelHeight / 2.0f, tunnelWidth / 2.0f)
        );
        
        // Spawn bounds not used when starting with 0 particles, but set to source area
        spawnBounds = Bounds3(
            Vec3(-tunnelLength / 2.0f - 1.0f, -tunnelHeight / 2.0f, -tunnelWidth / 2.0f),
            Vec3(-tunnelLength / 2.0f + 1.0f, tunnelHeight / 2.0f, tunnelWidth / 2.0f)
        );
        
        // Source at negative X end (inlet) - covers the cross-section
        float sourceThickness = 1.0f;  // Thin source plane
        Vec3 sourceCenter = Vec3(-tunnelLength / 2.0f, 0.0f, 0.0f);
        Vec3 sourceHalfExtents = Vec3(sourceThickness / 2.0f, tunnelHeight / 2.0f, tunnelWidth / 2.0f);
        sourceBounds = OBB(Transform(sourceCenter, Quat(Identity)), sourceHalfExtents);
        sourceSpeed = 20.0f;  // Particles shoot into tunnel at this speed
        sourceDensity = 1.0f;
        
        // Sink at positive X end (outlet) - covers the cross-section
        float sinkThickness = 1.0f;  // Thin sink plane
        Vec3 sinkCenter = Vec3(tunnelLength / 2.0f, 0.0f, 0.0f);
        Vec3 sinkHalfExtents = Vec3(sinkThickness / 2.0f, tunnelHeight / 2.0f, tunnelWidth / 2.0f);
        sinkBounds = OBB(Transform(sinkCenter, Quat(Identity)), sinkHalfExtents);
        
        // Camera positioned to view the tunnel from the side
        cameraPos = Vec3(0.0f, 10.0f, 20.0f);
        cameraLookAt = Vec3(0.0f, 0.0f, 0.0f);
        cameraNear = 0.1f;
        cameraFar = 1000.0f;
        
        // No mesh walls (user will add obstacles later)
        meshName = "";
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
    void handleSourceAndSink();

    FluidDemoDescriptor demoDesc;
    std::string customName = "Fluid Demo";
    int lastInitializedParticleCount = -1;

    // Point rendering
    PointRenderer* particleRenderer = nullptr;
    
    // Surface rendering
    SurfaceRenderer* surfaceRenderer = nullptr;
    MarchingCubesRenderer* marchingCubesRenderer = nullptr;
    std::shared_ptr<MarchingCubesSurface> marchingCubesSurface = nullptr;
    FluidRenderMode renderMode = FluidRenderMode::Particles;
    
    // CUDA resources
    cudaGraphicsResource* cudaVboResource = nullptr;
    std::shared_ptr<FluidDeviceData> deviceData = nullptr;
    std::shared_ptr<CudaMeshes> meshes = nullptr;
    std::shared_ptr<CudaHash> hash = nullptr;

    std::vector<Vec3> sourceParticlePos;
    std::vector<Vec3> sourceParticleVel;

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

    // Debug
    bool showDebugParticles = false;
};
