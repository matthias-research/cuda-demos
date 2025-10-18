#include "ParticleDemo.h"
#include <imgui.h>

// Forward declare CUDA kernel launcher
extern "C" void launchParticleKernel(uchar4* ptr, unsigned int width, unsigned int height, float time);

void ParticleDemo::update(float deltaTime) {
    animTime += deltaTime * animSpeed;
}

void ParticleDemo::render(uchar4* d_out, int width, int height) {
    launchParticleKernel(d_out, width, height, animTime);
}

void ParticleDemo::renderUI() {
    ImGui::SliderFloat("Animation Speed", &animSpeed, 0.0f, 5.0f);
    if (ImGui::Button("Reset Animation")) {
        animTime = 0.0f;
    }
}

void ParticleDemo::reset() {
    animTime = 0.0f;
    animSpeed = 1.0f;
}

