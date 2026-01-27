#include "BallsDemo.h"
#include "RenderUtils.h"
#include "BVH.h"
#include <imgui.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

BallsDemo::BallsDemo(const BallsDemoDescriptor& desc) {
    demoDesc = desc;
    
    // Initialize point renderer with ball layout (14 floats per point)
    PointRenderer::AttribLayout layout;
    layout.strideFloats = 14;
    layout.posOffset = 0;
    layout.radiusOffset = 3;
    layout.colorOffset = 4;
    layout.quatOffset = 7;
    
    pointRenderer = new PointRenderer();
    pointRenderer->init(demoDesc.numBalls, layout);
    
    // Initialize mesh renderer
    meshRenderer = new Renderer();
    meshRenderer->init();
    
    // Scene will be loaded on demand (lazy loading)
    scene = new Scene();
    
    // Initialize skybox
    skybox = new Skybox();
    if (!skybox->loadFromBMP("assets/skybox.bmp")) {
        delete skybox;
        skybox = nullptr;
        showSkybox = false;
    }
    
    paused = true;
    
    initBalls();
    
    // Load ball texture
    if (!demoDesc.ballTextureName.empty()) {
        ballTexture = RenderUtils::loadTexture("assets/" + demoDesc.ballTextureName);
        useTextureMode = (ballTexture != 0);
    } else {
        ballTexture = RenderUtils::createCheckerTexture();
    }
}

void BallsDemo::ensureSceneLoaded() {
    if (!sceneLoaded) {
        printf("Loading scene: %s\n", demoDesc.fileName.c_str());
        if (scene->load("assets/" + demoDesc.fileName)) {
            showScene = true;
            printf("Scene loaded successfully!\n");
        } else {
            printf("Failed to load scene: %s\n", demoDesc.fileName.c_str());
        }
        sceneLoaded = true;
    }
    
    // Reinitialize CUDA if ball count changed
    if (lastInitializedBallCount != demoDesc.numBalls) {
        if (cudaVboResource) {
            cleanupCudaPhysics(cudaVboResource);
        }
        initCudaPhysics(pointRenderer->getVBO(), &cudaVboResource, scene);
        lastInitializedBallCount = demoDesc.numBalls;
    }
}

BallsDemo::~BallsDemo() {
    if (cudaVboResource) {
        cleanupCudaPhysics(cudaVboResource);
    }
    if (pointRenderer) {
        pointRenderer->cleanup();
        delete pointRenderer;
    }
    if (scene) {
        scene->cleanup();
        delete scene;
    }
    if (meshRenderer) {
        meshRenderer->cleanup();
        delete meshRenderer;
    }
    if (skybox) {
        skybox->cleanup();
        delete skybox;
    }
    if (shadowFBO) glDeleteFramebuffers(1, &shadowFBO);
    if (shadowTexture) glDeleteTextures(1, &shadowTexture);
    if (ballTexture) glDeleteTextures(1, &ballTexture);
}

void BallsDemo::initBalls() {
    GLuint vbo = pointRenderer->getVBO();
    if (vbo != 0) {
        initCudaPhysics(vbo, &cudaVboResource, scene);
    }
}


void BallsDemo::initShadowBuffer(int width, int height) {
    if (shadowFBO != 0 && shadowWidth == width && shadowHeight == height) {
        return;
    }
    
    if (shadowFBO != 0) {
        glDeleteFramebuffers(1, &shadowFBO);
        glDeleteTextures(1, &shadowTexture);
    }
    
    shadowWidth = width;
    shadowHeight = height;
    
    // Create FBO
    glGenFramebuffers(1, &shadowFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    
    // Create color texture
    glGenTextures(1, &shadowTexture);
    glBindTexture(GL_TEXTURE_2D, shadowTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, shadowTexture, 0);
    
    // Create depth renderbuffer
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Shadow framebuffer is not complete!" << std::endl;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BallsDemo::renderShadows(int width, int height) {
    if (!camera || !pointRenderer) return;
    
    initShadowBuffer(width, height);
    
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Build light view-projection matrix
    Vec3 lightDirection = lightDir.normalized();
    float roomDiag = (demoDesc.sceneBounds.maximum - demoDesc.sceneBounds.minimum).magnitude();
    Vec3 lightPos = -lightDirection * (roomDiag * 2.0f);
    Vec3 target(0.0f, 0.0f, 0.0f);
    
    Vec3 forward = (target - lightPos).normalized();
    Vec3 right = Vec3(0, 1, 0).cross(forward).normalized();
    Vec3 up = forward.cross(right);
    
    float view[16];
    view[0] = right.x;    view[4] = right.y;    view[8] = right.z;     view[12] = -right.dot(lightPos);
    view[1] = up.x;       view[5] = up.y;       view[9] = up.z;        view[13] = -up.dot(lightPos);
    view[2] = -forward.x; view[6] = -forward.y; view[10] = -forward.z; view[14] = forward.dot(lightPos);
    view[3] = 0;          view[7] = 0;          view[11] = 0;          view[15] = 1;
    
    float halfSize = roomDiag * 0.6f;
    float near = 0.1f;
    float far = roomDiag * 4.0f;
    
    float projection[16];
    RenderUtils::buildOrthographicMatrix(projection, halfSize, near, far);
    
    // Combine into light view-projection matrix
    float lightViewProj[16];
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            lightViewProj[col * 4 + row] = 0.0f;
            for (int k = 0; k < 4; k++) {
                lightViewProj[col * 4 + row] += projection[k * 4 + row] * view[col * 4 + k];
            }
        }
    }
    
    pointRenderer->renderShadowPass(lightViewProj, demoDesc.numBalls, projection[0] * height);
    
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BallsDemo::update(float deltaTime) {
    // Ensure scene is loaded on first use
    ensureSceneLoaded();
    
    // Track FPS
    if (lastUpdateTime > 0.0f) {
        fps = 1.0f / deltaTime;
    }
    lastUpdateTime = deltaTime;
    
    // Update physics on GPU (only if not paused)
    if (!paused && cudaVboResource) {
        updateCudaPhysics(deltaTime, cudaVboResource);
    }
}

bool BallsDemo::raycast(const Vec3& orig, const Vec3& dir, float& minT) 
{
    // raycast against mesh
    minT = MaxFloat;
    float t;
    if (cudaRaycast(Ray(orig, dir), t))
    {
        minT = t;
    }

    // raycast against floor plane of the sceneBounds

    if (dir.y < -1e-6) {
        t = (demoDesc.sceneBounds.minimum.y - orig.y) / dir.y;
        if (t > 0 && t < minT) {
            minT = t;
        }
    }
    return minT < MaxFloat;
}

void BallsDemo::render(uchar4* d_out, int width, int height) {
    // Clear the PBO to prevent artifacts
    // Actual rendering happens in render3D() directly to screen
    cudaMemset(d_out, 0, width * height * sizeof(uchar4));
}


void BallsDemo::renderUI() {
    ImGui::Text("=== GPU-Accelerated Physics Demo ===");
    ImGui::Separator();
    
    if (paused) {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "[PAUSED] - Press 'P' to resume");
    } else {
        ImGui::Text("Press 'P' to pause simulation");
    }
    ImGui::Separator();
    
    ImGui::Text("Performance:");
    ImGui::Text("  FPS: %.1f", fps);
    ImGui::Text("  Frame Time: %.2f ms", lastUpdateTime * 1000.0f);
    ImGui::Text("  Ball Count: %d", demoDesc.numBalls);
    
    ImGui::Separator();
    ImGui::Text("Camera Controls:");
    ImGui::Text("  WASD: Move, Q/E: Up/Down");
    ImGui::Text("  Left Mouse: Orbit");
    ImGui::Text("  Middle Mouse: Pan");
    ImGui::Text("  Right Mouse: Rotate View");
    ImGui::Text("  Wheel: Zoom");
    
    ImGui::Separator();
    ImGui::Text("GPU Simulation Parameters:");
    
    bool ballCountChanged = false;
    if (ImGui::SliderInt("Ball Count##balls", &demoDesc.numBalls, 1000, 100000)) {
        ballCountChanged = true;
    }
    
    ImGui::SliderFloat("Gravity##balls", &demoDesc.gravity, 0.0f, 20.0f, "%.1f");
    ImGui::SliderFloat("Bounce##balls", &demoDesc.bounce, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Friction##balls", &demoDesc.friction, 0.8f, 1.0f, "%.3f");
    ImGui::Text("Scene Bounds: [%.0f, %.0f, %.0f] to [%.0f, %.0f, %.0f]",
        demoDesc.sceneBounds.minimum.x, demoDesc.sceneBounds.minimum.y, demoDesc.sceneBounds.minimum.z,
        demoDesc.sceneBounds.maximum.x, demoDesc.sceneBounds.maximum.y, demoDesc.sceneBounds.maximum.z);
    ImGui::Text("Spawn Bounds: [%.0f, %.0f, %.0f] to [%.0f, %.0f, %.0f]",
        demoDesc.ballsBounds.minimum.x, demoDesc.ballsBounds.minimum.y, demoDesc.ballsBounds.minimum.z,
        demoDesc.ballsBounds.maximum.x, demoDesc.ballsBounds.maximum.y, demoDesc.ballsBounds.maximum.z);
    
    ImGui::Separator();
    ImGui::Text("Lighting (Directional):");
    float lightAzimuthDegrees = demoDesc.lightAzimuth * 180.0f / 3.14159265f;
    float lightElevationDegrees = demoDesc.lightElevation * 180.0f / 3.14159265f;
    if (ImGui::SliderFloat("Light Azimuth (deg)##balls", &lightAzimuthDegrees, 0.0f, 360.0f)) {
        demoDesc.lightAzimuth = lightAzimuthDegrees * 3.14159265f / 180.0f;
    }
    if (ImGui::SliderFloat("Light Elevation (deg)##balls", &lightElevationDegrees, 0.0f, 180.0f)) {
        demoDesc.lightElevation = lightElevationDegrees * 3.14159265f / 180.0f;
    }
    // Convert spherical coordinates to direction vector
    float sinElev = sinf(demoDesc.lightElevation);
    float cosElev = cosf(demoDesc.lightElevation);
    float sinAzim = sinf(demoDesc.lightAzimuth);
    float cosAzim = cosf(demoDesc.lightAzimuth);
    lightDir = Vec3(sinElev * cosAzim, cosElev, sinElev * sinAzim).normalized();
    
    ImGui::Separator();
    ImGui::Text("Sun Direction (Ball Shadows):");
    float sunAzimuthDegrees = demoDesc.sunAzimuth * 180.0f / 3.14159265f;
    float sunElevationDegrees = demoDesc.sunElevation * 180.0f / 3.14159265f;
    if (ImGui::SliderFloat("Sun Azimuth (deg)##balls", &sunAzimuthDegrees, 0.0f, 360.0f)) {
        demoDesc.sunAzimuth = sunAzimuthDegrees * 3.14159265f / 180.0f;
    }
    if (ImGui::SliderFloat("Sun Elevation (deg)##balls", &sunElevationDegrees, 0.0f, 180.0f)) {
        demoDesc.sunElevation = sunElevationDegrees * 3.14159265f / 180.0f;
    }
    // Convert spherical coordinates to sun direction vector
    float sunSinElev = sinf(demoDesc.sunElevation);
    float sunCosElev = cosf(demoDesc.sunElevation);
    float sunSinAzim = sinf(demoDesc.sunAzimuth);
    float sunCosAzim = cosf(demoDesc.sunAzimuth);
    demoDesc.sunDirection = Vec3(sunSinElev * sunCosAzim, sunCosElev, sunSinElev * sunSinAzim).normalized();
    
    ImGui::Separator();
    ImGui::Text("Static Scene:");
    if (scene) {
        ImGui::Text("  Meshes loaded: %zu", scene->getMeshCount());
        ImGui::Checkbox("Show Scene##balls", &showScene);
        if (meshRenderer) {
            ImGui::Checkbox("Use Baked Lighting##balls", &demoDesc.useBakedLighting);
            if (!demoDesc.useBakedLighting) {
                ImGui::SliderFloat("Mesh Ambient##balls", &demoDesc.meshAmbient, 0.0f, 1.0f);
                meshRenderer->getMaterial().ambientStrength = demoDesc.meshAmbient;
                ImGui::SliderFloat("Mesh Specular##balls", &meshRenderer->getMaterial().specularStrength, 0.0f, 2.0f);
            }
        }
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No mesh loaded");
        ImGui::Text("Place a .glb file in assets/ folder");
    }
    
    ImGui::Separator();
    ImGui::Text("Skybox:");
    if (skybox) {
        ImGui::Checkbox("Show Skybox##balls", &showSkybox);
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No skybox loaded");
        ImGui::Text("Place skybox.bmp in assets/ folder");
    }
    
    ImGui::Separator();
    if (ImGui::Button("Reset View##balls", ImVec2(200, 0))) {
        if (camera) camera->resetView();
    }
    
    // Handle ball count change
    if (ballCountChanged) {
        if (cudaVboResource) {
            cleanupCudaPhysics(cudaVboResource);
            cudaVboResource = nullptr;
        }
        
        pointRenderer->resize(demoDesc.numBalls);
        initBalls();
    }
    
    ImGui::Separator();
    ImGui::Text("Ball Appearance:");
    ImGui::Checkbox("Use Texture Mode##balls", &useTextureMode);
    if (useTextureMode) {
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Texture mode (ready for texture asset)");
    } else {
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.5f, 1.0f), "Beach ball pattern mode");
    }
}

void BallsDemo::reset() {
    // Reset to descriptor defaults
    // (descriptor values are set in setupCityScene(), setupBunnyScene(), etc.)
    // Currently just reinitializing with current descriptor values
    
    // Reinitialize CUDA physics
    if (cudaVboResource) {
        cleanupCudaPhysics(cudaVboResource);
        cudaVboResource = nullptr;
    }
    initBalls();
    
    // Reset camera to descriptor settings
    if (camera) {
        camera->lookAt(demoDesc.cameraPos, demoDesc.cameraLookAt);
    }
}

void BallsDemo::onKeyPress(unsigned char key) {
    if (key == 'p' || key == 'P') {
        paused = !paused;
    }
}

void BallsDemo::render3D(int width, int height) {
    if (!camera || !pointRenderer) return;
    
    camera->nearClip = demoDesc.cameraNear;
    camera->farClip = demoDesc.cameraFar;
    
    glViewport(0, 0, width, height);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (skybox && showSkybox) {
        skybox->render(camera);
    }
    
    // Render balls using PointRenderer
    PointRenderer::Mode mode = useTextureMode ? PointRenderer::Mode::Ball : PointRenderer::Mode::Ball;
    pointRenderer->render(camera, demoDesc.numBalls, mode, 
                         useTextureMode ? ballTexture : 0, lightDir, width, height);
    
    // Render static scene meshes
    if (showScene && scene && meshRenderer) {
        Vec3 lightDirection = lightDir.normalized();
        meshRenderer->getLight().x = lightDirection.x;
        meshRenderer->getLight().y = lightDirection.y;
        meshRenderer->getLight().z = lightDirection.z;
        
        glDisable(GL_CULL_FACE);
        meshRenderer->setUseBakedLighting(demoDesc.useBakedLighting);
        
        Renderer::ShadowMap shadowMapData;
        shadowMapData.texture = shadowTexture;
        shadowMapData.bias = 0.005f;
        
        // Build light space matrix for shadows
        float roomDiag = (demoDesc.sceneBounds.maximum - demoDesc.sceneBounds.minimum).magnitude();
        Vec3 lightPos = -lightDirection * (roomDiag * 2.0f);
        Vec3 target(0.0f, 0.0f, 0.0f);
        Vec3 forward = (target - lightPos).normalized();
        Vec3 right = Vec3(0, 1, 0).cross(forward).normalized();
        Vec3 up = forward.cross(right);
        
        float lightView[16];
        lightView[0] = right.x;    lightView[4] = right.y;    lightView[8] = right.z;     lightView[12] = -right.dot(lightPos);
        lightView[1] = up.x;       lightView[5] = up.y;       lightView[9] = up.z;        lightView[13] = -up.dot(lightPos);
        lightView[2] = -forward.x; lightView[6] = -forward.y; lightView[10] = -forward.z; lightView[14] = forward.dot(lightPos);
        lightView[3] = 0;          lightView[7] = 0;          lightView[11] = 0;          lightView[15] = 1;
        
        float halfSize = roomDiag * 0.6f;
        float nearClip = 0.1f;
        float farClip = roomDiag * 4.0f;
        float lightProj[16];
        RenderUtils::buildOrthographicMatrix(lightProj, halfSize, nearClip, farClip);
        
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                shadowMapData.lightMatrix[col * 4 + row] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    shadowMapData.lightMatrix[col * 4 + row] += lightProj[k * 4 + row] * lightView[col * 4 + k];
                }
            }
        }
        
        for (const Mesh* mesh : scene->getMeshes()) {
            meshRenderer->renderMesh(*mesh, camera, width, height, 1.0f, nullptr, 0.0f, useShadows ? &shadowMapData : nullptr);
        }
    }
    
    glDisable(GL_DEPTH_TEST);
    
    if (useShadows) {
        renderShadows(width, height);
    }
}
