#include "WaveDemo.h"
#include <imgui.h>

// Forward declare CUDA kernel launcher
extern "C" void launchWaveKernel(uchar4* ptr, unsigned int width, unsigned int height, float time);

void WaveDemo::update(float deltaTime) {
    animTime += deltaTime * animSpeed;
}

void WaveDemo::render(uchar4* d_out, int width, int height) {
    launchWaveKernel(d_out, width, height, animTime);
}

void WaveDemo::renderUI() {
    ImGui::SliderFloat("Animation Speed", &animSpeed, 0.0f, 5.0f);
    if (ImGui::Button("Reset Animation")) {
        animTime = 0.0f;
    }
}

void WaveDemo::reset() {
    animTime = 0.0f;
    animSpeed = 1.0f;
}

