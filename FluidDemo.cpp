#include "FluidDemo.h"
#include "RenderUtils.h"
#include <imgui.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>


FluidDemo::FluidDemo(const FluidDemoDescriptor& desc) {
    demoDesc = desc;

    // Initialize point renderer with particle layout (8 floats per particle)
    PointRenderer::AttribLayout layout;
    layout.strideFloats = 8;
    layout.posOffset = 0;
    layout.radiusOffset = -1;      // Using uniform radius
    layout.colorOffset = 3;
    layout.quatOffset = -1;
    layout.lifetimeOffset = 6;

    particleRenderer = new PointRenderer();
    particleRenderer->init(demoDesc.numParticles, layout);

    // Initialize surface renderer (screen-space)
    surfaceRenderer = new SurfaceRenderer();
    surfaceRenderer->init(demoDesc.numParticles, layout);

    // Initialize marching cubes renderer
    marchingCubesRenderer = new MarchingCubesRenderer();
    marchingCubesRenderer->init();

    // Initialize marching cubes surface (uses kernel radius for grid spacing)
    marchingCubesSurface = std::make_shared<MarchingCubesSurface>();
    marchingCubesSurface->initialize(demoDesc.numParticles, demoDesc.kernelRadius, true);

    // Initialize mesh renderer
    meshRenderer = new Renderer();
    meshRenderer->init();

    // Scene will be loaded on demand
    scene = new Scene();

    // Initialize skybox
    skybox = new Skybox();
    if (!skybox->loadFromBMP("assets/skybox.bmp")) {
        delete skybox;
        skybox = nullptr;
        showSkybox = false;
    }

    paused = true;
    initParticles();
}


FluidDemo::~FluidDemo() {
    if (cudaVboResource) {
        cleanupCudaPhysics(cudaVboResource);
    }
    if (particleRenderer) {
        particleRenderer->cleanup();
        delete particleRenderer;
    }
    if (surfaceRenderer) {
        surfaceRenderer->cleanup();
        delete surfaceRenderer;
    }
    if (marchingCubesRenderer) {
        marchingCubesRenderer->cleanup();
        delete marchingCubesRenderer;
    }
    if (marchingCubesSurface) {
        marchingCubesSurface->free();
        marchingCubesSurface.reset();
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
}


void FluidDemo::ensureSceneLoaded() {
    if (!sceneLoaded) {
        if (!demoDesc.meshName.empty()) {
            printf("Loading scene: %s\n", demoDesc.meshName.c_str());
            if (scene->load("assets/" + demoDesc.meshName)) {
                showScene = true;
                printf("Scene loaded successfully!\n");
            } else {
                printf("Failed to load scene: %s\n", demoDesc.meshName.c_str());
            }
        }
        sceneLoaded = true;
    }

    // Reinitialize CUDA if particle count changed
    if (lastInitializedParticleCount != demoDesc.numParticles) {
        if (cudaVboResource) {
            cleanupCudaPhysics(cudaVboResource);
        }
        initCudaPhysics(particleRenderer->getVBO(), &cudaVboResource, scene);
        lastInitializedParticleCount = demoDesc.numParticles;
    }
}


void FluidDemo::initParticles() {
    GLuint vbo = particleRenderer->getVBO();
    if (vbo != 0) {
        initCudaPhysics(vbo, &cudaVboResource, scene);
    }
}


void FluidDemo::update(float deltaTime) {
    ensureSceneLoaded();

    if (lastUpdateTime > 0.0f) {
        fps = 1.0f / deltaTime;
    }
    lastUpdateTime = deltaTime;

    if (!paused && cudaVboResource) {
        updateCudaPhysics(deltaTime, cudaVboResource);
    }
}


bool FluidDemo::raycast(const Vec3& orig, const Vec3& dir, float& minT) {
    minT = MaxFloat;
    float t;
    if (cudaRaycast(Ray(orig, dir), t)) {
        minT = t;
    }

    // Raycast against floor
    if (dir.y < -1e-6) {
        t = (demoDesc.sceneBounds.minimum.y - orig.y) / dir.y;
        if (t > 0 && t < minT) {
            minT = t;
        }
    }
    return minT < MaxFloat;
}


void FluidDemo::render(uchar4* d_out, int width, int height) {
    cudaMemset(d_out, 0, width * height * sizeof(uchar4));
}


void FluidDemo::renderUI() {
    ImGui::Text("=== Fluid Particle Demo ===");
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
    ImGui::Text("  Particle Count: %d", demoDesc.numParticles);

    ImGui::Separator();
    ImGui::Text("Camera Controls:");
    ImGui::Text("  WASD: Move, Q/E: Up/Down");
    ImGui::Text("  Mouse: Orbit/Pan/Rotate");
    ImGui::Text("  Wheel: Zoom");

    ImGui::Separator();
    ImGui::Text("Simulation Parameters:");

    bool particleCountChanged = false;
    if (ImGui::SliderInt("Particle Count##fluid", &demoDesc.numParticles, 1, 200000)) {
        particleCountChanged = true;
    }

    ImGui::SliderFloat("Gravity##fluid", &demoDesc.gravity, 0.0f, 20.0f, "%.1f");
    ImGui::SliderFloat("Particle Radius##fluid", &demoDesc.particleRadius, 0.1f, 1.0f, "%.2f");
    ImGui::SliderFloat("Max Velocity##fluid", &demoDesc.maxVelocity, 10.0f, 200.0f, "%.1f");

    ImGui::Separator();
    ImGui::Text("Lighting:");
    float lightAzimuthDegrees = demoDesc.lightAzimuth * 180.0f / 3.14159265f;
    float lightElevationDegrees = demoDesc.lightElevation * 180.0f / 3.14159265f;
    if (ImGui::SliderFloat("Light Azimuth##fluid", &lightAzimuthDegrees, 0.0f, 360.0f)) {
        demoDesc.lightAzimuth = lightAzimuthDegrees * 3.14159265f / 180.0f;
    }
    if (ImGui::SliderFloat("Light Elevation##fluid", &lightElevationDegrees, 0.0f, 180.0f)) {
        demoDesc.lightElevation = lightElevationDegrees * 3.14159265f / 180.0f;
    }
    float sinElev = sinf(demoDesc.lightElevation);
    float cosElev = cosf(demoDesc.lightElevation);
    float sinAzim = sinf(demoDesc.lightAzimuth);
    float cosAzim = cosf(demoDesc.lightAzimuth);
    lightDir = Vec3(sinElev * cosAzim, cosElev, sinElev * sinAzim).normalized();

    ImGui::Separator();
    ImGui::Text("Render Mode:");
    int currentMode = static_cast<int>(renderMode);
    const char* modeNames[] = { "Particles", "Screen Space Surface", "Marching Cubes" };
    if (ImGui::Combo("##RenderMode", &currentMode, modeNames, 3)) {
        renderMode = static_cast<FluidRenderMode>(currentMode);
    }

    // Show screen space surface parameters when in that mode
    if (renderMode == FluidRenderMode::ScreenSpaceSurface && surfaceRenderer) {
        SurfaceRenderer::FluidStyle& style = surfaceRenderer->getStyle();
        
        ImGui::Separator();
        ImGui::Text("Curvature Flow Smoothing:");
        ImGui::SliderInt("Iterations##ss", &style.smoothingIterations, 0, 200);
        ImGui::SliderFloat("Timestep (dt)##ss", &style.smoothingDt, 0.0001f, 0.01f, "%.4f");
        ImGui::SliderFloat("Z Contribution##ss", &style.smoothingZContrib, 0.0f, 50.0f, "%.1f");
        
        ImGui::Separator();
        ImGui::Text("Particle Rendering:");
        ImGui::SliderFloat("Depth Scale##ss", &style.particleScale, 0.5f, 10.0f, "%.1f");
        ImGui::SliderFloat("Thickness Scale##ss", &style.thicknessScale, 0.5f, 10.0f, "%.1f");
        
        ImGui::Separator();
        ImGui::Text("Fluid Appearance:");
        ImGui::ColorEdit3("Absorption##ss", &style.absorptionCoeff.x);
        ImGui::SliderFloat("Absorption Scale##ss", &style.absorptionScale, 0.1f, 5.0f, "%.1f");
        
        ImGui::Separator();
        ImGui::Text("Lighting:");
        ImGui::SliderFloat("Specular Power##ss", &style.specularPower, 1.0f, 256.0f, "%.0f");
        ImGui::SliderFloat("Fresnel Power##ss", &style.fresnelPower, 1.0f, 10.0f, "%.1f");
        ImGui::SliderFloat("Ambient##ss", &style.ambientStrength, 0.0f, 1.0f, "%.2f");
    }

    // Show marching cubes stats when in that mode
    if (renderMode == FluidRenderMode::MarchingCubes && marchingCubesSurface) {
        ImGui::Text("  Cubes: %d", marchingCubesSurface->getNumCubes());
        ImGui::Text("  Vertices: %d", marchingCubesSurface->getNumVertices());
        ImGui::Text("  Triangles: %d", marchingCubesSurface->getNumTriangles());
        
        GLuint vbo = marchingCubesSurface->getVerticesVbo();
        GLuint ibo = marchingCubesSurface->getTriIndicesIbo();
        ImGui::Text("  VBO: %u, IBO: %u", vbo, ibo);
        
        ImGui::Checkbox("Show Debug Particles##mc", &showDebugParticles);
    }

    ImGui::Separator();
    ImGui::Text("Scene:");
    if (scene && scene->getMeshCount() > 0) {
        ImGui::Text("  Meshes loaded: %zu", scene->getMeshCount());
        ImGui::Checkbox("Show Scene##fluid", &showScene);
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No mesh loaded");
    }

    if (skybox) {
        ImGui::Checkbox("Show Skybox##fluid", &showSkybox);
    }

    ImGui::Separator();
    if (ImGui::Button("Reset View##fluid", ImVec2(200, 0))) {
        if (camera) camera->resetView();
    }

    // Handle particle count change
    if (particleCountChanged) {
        if (cudaVboResource) {
            cleanupCudaPhysics(cudaVboResource);
            cudaVboResource = nullptr;
        }
        particleRenderer->resize(demoDesc.numParticles);
        
        // Reinitialize marching cubes surface
        if (marchingCubesSurface) {
            marchingCubesSurface->free();
            marchingCubesSurface->initialize(demoDesc.numParticles, demoDesc.kernelRadius, true);
        }
        
        initParticles();
    }
}


void FluidDemo::reset() {
    if (cudaVboResource) {
        cleanupCudaPhysics(cudaVboResource);
        cudaVboResource = nullptr;
    }
    initParticles();

    if (camera) {
        camera->lookAt(demoDesc.cameraPos, demoDesc.cameraLookAt);
    }
}


void FluidDemo::onKeyPress(unsigned char key) {
    if (key == 'p' || key == 'P') {
        paused = !paused;
    }
}


void FluidDemo::render3D(int width, int height) {
    if (!camera || !particleRenderer) return;

    camera->nearClip = demoDesc.cameraNear;
    camera->farClip = demoDesc.cameraFar;

    glViewport(0, 0, width, height);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (skybox && showSkybox) {
        skybox->render(camera);
    }

    // Render fluid based on selected mode
    switch (renderMode) {
        case FluidRenderMode::Particles:
            particleRenderer->render(camera, demoDesc.numParticles, PointRenderer::Mode::Particle,
                                     0, lightDir, width, height, demoDesc.particleRadius);
            break;

        case FluidRenderMode::ScreenSpaceSurface:
            if (surfaceRenderer) {
                surfaceRenderer->render(camera, particleRenderer->getVBO(), demoDesc.numParticles,
                                        demoDesc.particleRadius, lightDir, width, height);
            }
            break;

        case FluidRenderMode::MarchingCubes:
            if (marchingCubesRenderer && marchingCubesSurface) {
                marchingCubesRenderer->render(camera, marchingCubesSurface.get(), lightDir, width, height);
                
                // Debug: also render particles as small points to see if surface aligns
                if (showDebugParticles) {
                    glDepthMask(GL_FALSE);  // Don't write to depth buffer
                    particleRenderer->render(camera, demoDesc.numParticles, PointRenderer::Mode::Particle,
                                             0, lightDir, width, height, demoDesc.particleRadius * 0.3f);
                    glDepthMask(GL_TRUE);
                }
            }
            break;
    }

    // Render static scene meshes
    if (showScene && scene && meshRenderer) {
        Vec3 lightDirection = lightDir.normalized();
        meshRenderer->getLight().x = lightDirection.x;
        meshRenderer->getLight().y = lightDirection.y;
        meshRenderer->getLight().z = lightDirection.z;

        glDisable(GL_CULL_FACE);
        meshRenderer->getMaterial().ambientStrength = demoDesc.meshAmbient;

        for (const Mesh* mesh : scene->getMeshes()) {
            meshRenderer->renderMesh(*mesh, camera, width, height, 1.0f, nullptr, 0.0f, nullptr);
        }
    }

    glDisable(GL_DEPTH_TEST);
}
