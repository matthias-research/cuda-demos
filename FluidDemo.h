#pragma once

#include <string>
#include <memory>
#include "Vec.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "Scene.h"
#include "Demo.h"

struct FluidDemoDescriptor {
    int numParticles = 100000;
    float gravity = 9.8f;
    float viscosity = 0.1f;
    float restDensity = 1000.0f;
    float particleRadius = 0.2f;
    float kernelRadius = 0.4f;
    float gridSpacing = 0.4f;
    float timeStep = 0.005f;
    float boundaryDamping = 0.5f;
    float surfaceTension = 0.0f;
    float vorticityConfinement = 0.0f;
    float maxVelocity = 100.0f;

    std::string meshName = "";
 
    // Optionally, add scene/camera presets
    std::string sceneName = "default";
    // Camera presets
    Vec3 cameraPos = Vec3(0.0f, 20.0f, 50.0f);
    Vec3 cameraLookAt = Vec3(0.0f, 10.0f, 0.0f);
    float cameraNear = 0.1f;
    float cameraFar = 1000.0f;

    void setupSphereScene()
    {
        numParticles = 50000;
        gravity = 0.0f;
        viscosity = 0.1f;
        restDensity = 1000.0f;
        particleRadius = 0.2f;
        kernelRadius = 0.4f;
        timeStep = 0.005f;
        maxVelocity = 100.0f;

        sceneName = "sphere";

        cameraPos = Vec3(0.0f, 20.0f, 50.0f);
        cameraLookAt = Vec3(0.0f, 10.0f, 0.0f);
    }
};


struct FluidDeviceData;
class CudaMeshes;

class FluidDemo : public Demo {
public:
    FluidDemo() = default;
    ~FluidDemo() override = default;

    const char* getName() const override { return "Fluid Demo"; }
    bool is3D() const override { return true; }
    void update(float /*deltaTime*/) override {}
    void render(uchar4* /*d_out*/, int /*width*/, int /*height*/) override {}
    void renderUI() override {}
    void reset() override {}
    void render3D(int /*width*/, int /*height*/) override {}
    void setCamera(Camera* /*cam*/) override {}
    bool raycast(const Vec3& /*orig*/, const Vec3& /*dir*/, float& /*t*/) override { return false; }
    void onMouseClick(int /*button*/, int /*state*/, int /*x*/, int /*y*/) override {}
    void onMouseDrag(int /*x*/, int /*y*/) override {}
    void onMouseWheel(int /*wheel*/, int /*direction*/, int /*x*/, int /*y*/) override {}
    void onKeyPress(unsigned char /*key*/) override {}
    void onSpecialKey(int /*key*/) override {}

    // CUDA physics functions (implemented in FluidDemo.cu)
    void initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, Scene* scene);
    void updateCudaPhysics(float dt, cudaGraphicsResource* vboResource);
    void cleanupCudaPhysics(cudaGraphicsResource* vboResource);
    bool cudaRaycast(const Ray& ray, float& t);

private:
    std::shared_ptr<FluidDeviceData> deviceData = nullptr;
    std::shared_ptr<CudaMeshes> meshes = nullptr;
    FluidDemoDescriptor demoDesc;
    std::string customName = "Fluid Demo";
};
