#pragma once

#include "Demo.h"

class WaveDemo : public Demo {
private:
    float animTime = 0.0f;
    float animSpeed = 1.0f;

public:
    const char* getName() const override { return "Wave Simulation"; }
    void update(float deltaTime) override;
    void render(uchar4* d_out, int width, int height) override;
    void renderUI() override;
    void reset() override;
};

