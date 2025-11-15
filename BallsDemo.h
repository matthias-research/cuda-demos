#pragma once

#include "Demo.h"
#include "Camera.h"
#include "Vec.h"
#include <GL/glew.h>
#include <vector>

struct Ball {
    Vec3 pos;       // Position
    Vec3 vel;       // Velocity
    Quat quat;      // Quaternion orientation
    Vec3 angVel;    // Angular velocity
    Vec3 color;     // Color
    float radius;   // Size
};

class BallsDemo : public Demo {
private:
    std::vector<Ball> balls;
    int numBalls = 50;
    float gravity = 9.8f;
    float bounce = 0.85f;  // Coefficient of restitution
    float friction = 0.99f;
    
    // Simulation bounds
    float roomSize = 5.0f;
    
    // OpenGL resources
    GLuint vao, vbo;
    GLuint sphereShader;
    GLuint fbo, renderTexture;
    int fbWidth, fbHeight;
    
    // Camera (reference to main camera)
    Camera* camera = nullptr;
    
    // Lighting
    float lightPosX = 5.0f;
    float lightPosY = 8.0f;
    float lightPosZ = 5.0f;
    
    void initBalls();
    void updatePhysics(float deltaTime);
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

