#pragma once

#include <memory>
#include "Demo.h"
#include "Camera.h"
#include "Vec.h"
#include "Scene.h"
#include "Renderer.h"
#include "Skybox.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>

struct Ball {
    Vec3 pos;       // Position
    Vec3 vel;       // Velocity
    Quat quat;      // Quaternion orientation
    Vec3 angVel;    // Angular velocity
    Vec3 color;     // Color
    float radius;   // Size
};

struct BallsDemoDescriptor 
{
    int numBalls = 1000000;  
    float gravity = 9.8f;
    float bounce = 0.85f;  // Coefficient of restitution
    float friction = 1.0f; // no friction
    float terminalVelocity = 10.0f;
    float meshAmbient = 0.2f;  // Mesh ambient lighting strength
    float lightAzimuth = 1.2f;      // Light direction: angle around up axis (radians)
    float lightElevation = 1.1f;    // Light direction: angle from up axis (radians)
    float sunAzimuth = 1.2f;        // Sun direction for shadows: angle around up axis (radians)
    float sunElevation = 1.1f;      // Sun direction for shadows: angle from up axis (radians)
    
    // Ball size parameters
    float minRadius = 0.25f;
    float maxRadius = 0.25f;
    
    // Camera clipping planes
    float cameraNear = 10.0f;
    float cameraFar = 10000.0f;

    Vec3 sunDirection = Vec3(0.3f, 0.8f, 0.3f).normalized();

    // Camera position and orientation
    Vec3 cameraPos = Vec3(0.0f, 100.0f, 200.0f);
    Vec3 cameraLookAt = Vec3(0.0f, 0.0f, 0.0f);
    
    Bounds3 sceneBounds = Bounds3(Vec3(-600.0f, 0.0f, -310.0f), Vec3(600.0f, 600.0f, 310.0f));
    Bounds3 ballsBounds = Bounds3(Vec3(-200.0f, 250.0f, -100.0f), Vec3(200.0f, 300.0f, 100.0f));
    std::string fileName = "";
    std::string ballTextureName = "";  // Texture file for ball rendering
    bool useBakedLighting = false;

    void setupCityScene()
    {
        fileName = "city.glb";

        numBalls = 1000000;
        gravity = 9.8f;
        bounce = 0.85f;  // Coefficient of restitution
        friction = 1.0f; // no friction
        terminalVelocity = 10.0f;
        meshAmbient = 0.7f;
        lightAzimuth = 82.0f * 3.14159265f / 180.0f;
        lightElevation = 26.875f * 3.14159265f / 180.0f;
        sunAzimuth = 283.25f * 3.14159265f / 180.0f;
        sunElevation = 26.25f * 3.14159265f / 180.0f;

        // Ball size parameters
        minRadius = 0.5f;
        maxRadius = 0.5f;

        // Camera clipping planes
        cameraNear = 10.0f;
        cameraFar = 10000.0f;

        sunDirection = Vec3(0.3f, 0.8f, 0.3f).normalized();

        // sceneBounds = Bounds3(Vec3(-580.0f, 0.0f, -310.0f), Vec3(600.0f, 600.0f, 310.0f));
        // tower
        ballsBounds = Bounds3(Vec3(-200.0f, 0.0f, -50.0f), Vec3(-150.0f, 300.0f, 0.0f));
        // ballsBounds = Bounds3(Vec3(-580.0f, 100.0f, -300.0f), Vec3(500.0f, 300.0f, 300.0f));

        cameraPos = Vec3(0.0f, 100.0f, 200.0f);
        cameraLookAt = Vec3(0.0f, 50.0f, 0.0f);
        useBakedLighting = false;
    }

    void setupWembleyScene()
    {
        fileName = "wembley.glb";
        ballTextureName = "soccerBall.bmp";

//        numBalls = 70000000;
        numBalls = 1000;
        gravity = 9.8f;
        bounce = 0.85f;  // Coefficient of restitution
        friction = 1.0f; // no friction
        terminalVelocity = 10.0f;
        meshAmbient = 0.5f;
        lightAzimuth = 82.0f * 3.14159265f / 180.0f;
        lightElevation = 26.875f * 3.14159265f / 180.0f;
        sunAzimuth = 283.25f * 3.14159265f / 180.0f;
        sunElevation = 26.25f * 3.14159265f / 180.0f;

        // Ball size parameters
        minRadius = 0.11f;
        maxRadius = 0.11f;

        // Camera clipping planes
        cameraNear = 1.0f;
        cameraFar = 10000.0f;

        sunDirection = Vec3(0.3f, 0.8f, 0.3f).normalized();

        sceneBounds = Bounds3(Vec3(-200.0f, 0.0f, -200.0f), Vec3(200.0f, 600.0f, 200.0f));
        ballsBounds = Bounds3(Vec3(-20.0f, 10.0f, -20.0f), Vec3(20.0f, 300.0f, 20.0f));

        cameraPos = Vec3(0.0f, 100.0f, 200.0f);
        cameraLookAt = Vec3(0.0f, 50.0f, 0.0f);
        useBakedLighting = true;
    }

    void setupBunnyScene()
    {
        fileName = "bunny.glb";

        numBalls = 100;
        gravity = 9.8f;
        bounce = 0.85f;  // Coefficient of restitution
        friction = 1.0f; // no friction
        terminalVelocity = 10.0f;
        meshAmbient = 0.2f;
        lightAzimuth = 1.2f;
        lightElevation = 1.1f;
        sunAzimuth = 1.2f;
        sunElevation = 1.1f;

        // Ball size parameters
        minRadius = 0.5f;
        maxRadius = 1.0f;

        // Camera clipping planes
        cameraNear = 0.1f;
        cameraFar = 1000.0f;

        sunDirection = Vec3(-0.1f, -0.1f, -0.3f).normalized();

        sceneBounds = Bounds3(Vec3(-20.0f, 0.0f, -20.0f), Vec3(20.0f, 50.0f, 20.0f));
        ballsBounds = Bounds3(Vec3(-15.0f, 10.0f, -15.0f), Vec3(15.0f, 300.0f, 15.0f));
        
        cameraPos = Vec3(0.0f, 30.0f, 50.0f);
        cameraLookAt = Vec3(0.0f, 15.0f, 0.0f);
        useBakedLighting = false;

        sunDirection = Vec3(0.3f, 1.0f, 0.1f).normalized();
    }
};


struct BallsDeviceData;
class BVHBuilder;

class BallsDemo : public Demo {
private:

    BallsDemoDescriptor demoDesc;
    std::string customName = "3D Bouncing Balls";
    int lastInitializedBallCount = -1;  // Track for reinit detection
    GLuint vao, vbo;
    GLuint ballShader;
    GLuint ballShadowShader;
    GLuint shadowFBO, shadowTexture;
    int shadowWidth, shadowHeight;
    
    // CUDA resources
    cudaGraphicsResource* cudaVboResource = nullptr;
    
    // Performance tracking
    float lastUpdateTime = 0.0f;
    float fps = 0.0f;
    bool paused = false;
    
    // Camera (reference to main camera)
    Camera* camera = nullptr;
    std::shared_ptr<BallsDeviceData> deviceData = nullptr;
    std::shared_ptr<BVHBuilder> bvhBuilder = nullptr;

    // Lighting

    // To the light (OpenGl convention)
    Vec3 lightDir = Vec3(0.1f, 0.1f, 0.5f).normalized();
    bool useShadows = false;  // Toggle shadow mapping on/off
    bool useTextureMode = false;  // Toggle between texture mode and beach ball pattern
    GLuint ballTexture = 0;  // Texture for ball rendering

    // Static scene rendering
    Scene* scene = nullptr;
    Renderer* renderer = nullptr;
    bool showScene = false;
    bool sceneLoaded = false;  // Flag for lazy loading
    
    // Skybox
    Skybox* skybox = nullptr;
    bool showSkybox = true;
    
    void initBalls();
    void initGL();
    void initShaders();
    void initShadowBuffer(int width, int height);
    void renderShadows(int width, int height);
    void cleanupGL();
    GLuint loadTexture(const std::string& filename);

public:
    BallsDemo(const BallsDemoDescriptor& desc);
    ~BallsDemo();
    
    void setName(const std::string& name) { customName = name; }
    void ensureSceneLoaded();  // Load scene on first use

    // CUDA physics functions (implemented in BallsDemo.cu)
    void initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, Scene* scene);
    void updateCudaPhysics(float dt, cudaGraphicsResource* vboResource);
    void cleanupCudaPhysics(cudaGraphicsResource* vboResource);
    bool cudaRaycast(const Ray& ray, float& t);
    void exportBallsToFile(const std::string& filename);
    
    const char* getName() const override { return customName.c_str(); }
    bool is3D() const override { return true; }
    void setCamera(Camera* cam) override { camera = cam; }
    void applyCameraSettings() { if (camera) camera->lookAt(demoDesc.cameraPos, demoDesc.cameraLookAt); }
    bool raycast(const Vec3& orig, const Vec3& dir, float& t) override;
    void update(float deltaTime) override;
    void render(uchar4* d_out, int width, int height) override;
    void render3D(int width, int height) override;
    void renderUI() override;
    void reset() override;
    void onKeyPress(unsigned char key) override;
};

// Factory functions for BallsDeviceData (implemented in BallsDemo.cu)
BallsDeviceData* createBallsDeviceData();
void deleteBallsDeviceData(BallsDeviceData* data);

