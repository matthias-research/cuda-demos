#include "BallsDemo.h"
#include <imgui.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Simplified sphere shader using point sprites
const char* sphereVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aRadius;
layout (location = 2) in vec3 aColor;
layout (location = 3) in vec4 aQuat;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform vec3 viewPos;
uniform float pointScale;

out vec3 fragPos;
out float radius;
out vec3 sphereColor;
out vec3 eyePos;
out vec4 quat;

void main()
{
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    fragPos = aPos;
    radius = aRadius;
    sphereColor = aColor;
    eyePos = eyeSpacePos.xyz;
    quat = aQuat;
    
    // Calculate point size based on distance
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = aRadius * (pointScale / dist);
}
)";

const char* sphereFragmentShader = R"(
#version 330 core

const float PI = 3.14159265359;

uniform vec3 viewPos;
uniform vec3 lightPos;

in vec3 fragPos;
in float radius;
in vec3 sphereColor;
in vec3 eyePos;
in vec4 quat;

out vec4 fragColor;

// Quaternion rotation function
vec3 qtransform(vec4 q, vec3 v)
{
    return v + 2.0 * cross(cross(v, q.xyz) + q.w * v, q.xyz);
}

void main()
{
    // Calculate ray direction in view space
    vec3 viewDir = normalize(viewPos - fragPos);
    
    // Calculate sphere normal from point coordinate
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    
    // Discard fragments outside sphere
    if (r2 > 1.0)
        discard;
    
    // Calculate 3D position on sphere surface
    float h = sqrt(1.0 - r2);
    
    // Build tangent space (billboard aligned to view)
    vec3 a0 = normalize(fragPos - viewPos);
    vec3 a2 = vec3(0.0, 1.0, 0.0);
    vec3 a1 = normalize(cross(a0, a2));
    a2 = normalize(cross(a1, a0));
    
    // Calculate surface position
    vec3 localPos = radius * (coord.x * a1 + coord.y * a2 - h * a0);
    vec3 surfacePos = fragPos + localPos;
    vec3 normal = normalize(localPos);
    
    // Rotate normal by quaternion
    vec3 rotNormal = qtransform(quat, normal);
    
    // Create stripe pattern based on rotated normal
    float angle = atan(rotNormal.y, rotNormal.x);
    int segment = int((angle + PI) / PI * 6.0);
    
    vec3 color = sphereColor;
    if (segment % 2 == 0) {
        color = vec3(1.0, 1.0, 1.0) - color;
    }
    
    // Phong lighting
    vec3 lightDir = normalize(lightPos - surfacePos);
    float diffuse = max(0.0, dot(lightDir, normal));
    
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    
    float ambient = 0.2;
    
    vec3 finalColor = color * (ambient + diffuse * 0.8) + vec3(1.0) * specular * 0.5;
    fragColor = vec4(finalColor, 1.0);
}
)";

// Helper function to compile shader
static GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
    }
    
    return shader;
}

BallsDemo::BallsDemo() : vao(0), vbo(0), sphereShader(0), fbo(0), renderTexture(0),
                         fbWidth(0), fbHeight(0) {
    // Seed random number generator
    srand(static_cast<unsigned int>(time(nullptr)));
    
    initBalls();
    initGL();
}

BallsDemo::~BallsDemo() {
    cleanupGL();
}

void BallsDemo::initBalls() {
    balls.clear();
    balls.reserve(numBalls);
    
    for (int i = 0; i < numBalls; i++) {
        Ball ball;
        
        // Random position in 3D room
        ball.pos.x = -roomSize * 0.4f + (rand() % 1000) / 1000.0f * roomSize * 0.8f;
        ball.pos.y = 1.0f + (rand() % 1000) / 1000.0f * roomSize * 0.5f;
        ball.pos.z = -roomSize * 0.4f + (rand() % 1000) / 1000.0f * roomSize * 0.8f;
        
        // Random velocity
        ball.vel.x = -2.0f + (rand() % 1000) / 1000.0f * 4.0f;
        ball.vel.y = -2.0f + (rand() % 1000) / 1000.0f * 4.0f;
        ball.vel.z = -2.0f + (rand() % 1000) / 1000.0f * 4.0f;
        
        // Initialize quaternion to identity (no rotation)
        ball.quat = Quat(Identity);
        
        // Random angular velocity
        ball.angVel.x = -3.0f + (rand() % 1000) / 1000.0f * 6.0f;
        ball.angVel.y = -3.0f + (rand() % 1000) / 1000.0f * 6.0f;
        ball.angVel.z = -3.0f + (rand() % 1000) / 1000.0f * 6.0f;
        
        // Random color (vibrant colors)
        ball.color.x = 0.3f + (rand() % 70) / 100.0f;
        ball.color.y = 0.3f + (rand() % 70) / 100.0f;
        ball.color.z = 0.3f + (rand() % 70) / 100.0f;
        
        // Random size (4x larger than before)
        ball.radius = (0.1f + (rand() % 100) / 1000.0f * 0.3f) * 4.0f;
        
        balls.push_back(ball);
    }
}

void BallsDemo::initGL() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Radius attribute (location 1)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Color attribute (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(4 * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    // Quaternion attribute (location 3)
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(7 * sizeof(float)));
    glEnableVertexAttribArray(3);
    
    glBindVertexArray(0);
    
    initShaders();
}

void BallsDemo::initShaders() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, sphereVertexShader);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, sphereFragmentShader);
    
    sphereShader = glCreateProgram();
    glAttachShader(sphereShader, vertexShader);
    glAttachShader(sphereShader, fragmentShader);
    glLinkProgram(sphereShader);
    
    GLint success;
    glGetProgramiv(sphereShader, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(sphereShader, 512, nullptr, infoLog);
        std::cerr << "Shader linking failed:\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void BallsDemo::initFramebuffer(int width, int height) {
    if (fbo != 0 && fbWidth == width && fbHeight == height) {
        return;
    }
    
    if (fbo != 0) {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &renderTexture);
    }
    
    fbWidth = width;
    fbHeight = height;
    
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    
    glGenTextures(1, &renderTexture);
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTexture, 0);
    
    GLuint rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer is not complete!" << std::endl;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void BallsDemo::cleanupGL() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (sphereShader) glDeleteProgram(sphereShader);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (renderTexture) glDeleteTextures(1, &renderTexture);
}

void BallsDemo::updatePhysics(float deltaTime) {
    float halfRoom = roomSize * 0.5f;
    
    for (auto& ball : balls) {
        // Apply gravity
        ball.vel.y -= gravity * deltaTime;
        
        // Update position
        ball.pos += ball.vel * deltaTime;
        
        // Update orientation using angular velocity - use the built-in method
        ball.quat = ball.quat.rotateLinear(ball.quat, ball.angVel * deltaTime);
        
        // Apply friction
        ball.vel *= friction;
        ball.angVel *= friction;
        
        // Bounce off walls (X axis)
        if (ball.pos.x - ball.radius < -halfRoom) {
            ball.pos.x = -halfRoom + ball.radius;
            ball.vel.x = -ball.vel.x * bounce;
            // Collision adds spin (cross product)
            ball.angVel += Vec3(0, ball.vel.z, -ball.vel.y) * 0.5f;
        }
        if (ball.pos.x + ball.radius > halfRoom) {
            ball.pos.x = halfRoom - ball.radius;
            ball.vel.x = -ball.vel.x * bounce;
            ball.angVel += Vec3(0, -ball.vel.z, ball.vel.y) * 0.5f;
        }
        
        // Bounce off floor and ceiling (Y axis)
        if (ball.pos.y - ball.radius < 0) {
            ball.pos.y = ball.radius;
            ball.vel.y = -ball.vel.y * bounce;
            // Floor collision adds spin
            ball.angVel += Vec3(ball.vel.z, 0, -ball.vel.x) * 0.5f;
        }
        if (ball.pos.y + ball.radius > roomSize) {
            ball.pos.y = roomSize - ball.radius;
            ball.vel.y = -ball.vel.y * bounce;
            ball.angVel += Vec3(-ball.vel.z, 0, ball.vel.x) * 0.5f;
        }
        
        // Bounce off front/back walls (Z axis)
        if (ball.pos.z - ball.radius < -halfRoom) {
            ball.pos.z = -halfRoom + ball.radius;
            ball.vel.z = -ball.vel.z * bounce;
            ball.angVel += Vec3(-ball.vel.y, ball.vel.x, 0) * 0.5f;
        }
        if (ball.pos.z + ball.radius > halfRoom) {
            ball.pos.z = halfRoom - ball.radius;
            ball.vel.z = -ball.vel.z * bounce;
            ball.angVel += Vec3(ball.vel.y, -ball.vel.x, 0) * 0.5f;
        }
    }
    
    // Ball-to-ball collision detection
    for (size_t i = 0; i < balls.size(); i++) {
        for (size_t j = i + 1; j < balls.size(); j++) {
            Ball& b1 = balls[i];
            Ball& b2 = balls[j];
            
            Vec3 delta = b2.pos - b1.pos;
            float dist = delta.magnitude();
            float minDist = b1.radius + b2.radius;
            
            if (dist < minDist && dist > 0.001f) {
                // Normalize to get collision normal
                Vec3 normal = delta / dist;
                
                // Separate balls
                float overlap = minDist - dist;
                b1.pos -= normal * (overlap * 0.5f);
                b2.pos += normal * (overlap * 0.5f);
                
                // Calculate relative velocity
                Vec3 relVel = b2.vel - b1.vel;
                
                // Velocity along collision normal
                float dvn = relVel.dot(normal);
                
                // Only resolve if balls are moving towards each other
                if (dvn < 0) {
                    // Elastic collision (simplified, assuming equal mass)
                    float impulse = dvn * bounce;
                    Vec3 impulseVec = normal * impulse;
                    b1.vel += impulseVec;
                    b2.vel -= impulseVec;
                    
                    // Add spin from collision (torque = r × F)
                    float spinFactor = 0.3f;
                    Vec3 tangent = relVel - normal * dvn;
                    b1.angVel += tangent.cross(normal) * spinFactor;
                    b2.angVel -= tangent.cross(normal) * spinFactor;
                }
            }
        }
    }
}

void BallsDemo::update(float deltaTime) {
    updatePhysics(deltaTime);
}

bool BallsDemo::raycast(const Vec3& orig, const Vec3& dir, float& t) {
    // Raycast against floor plane
    if (dir.y < -0.001f) {
        float tPlane = -orig.y / dir.y;
        if (tPlane > 0) {
            t = tPlane;
            return true;
        }
    }
    return false;
}

void BallsDemo::render(uchar4* d_out, int width, int height) {
    if (!camera) return;
    
    initFramebuffer(width, height);
    
    // Render to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Enable point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    glUseProgram(sphereShader);
    
    // Set up matrices (same as BoxesDemo)
    float model[16], view[16], projection[16];
    for (int i = 0; i < 16; i++) model[i] = 0.0f;
    model[0] = model[5] = model[10] = model[15] = 1.0f;
    
    Vec3 camPos = camera->pos;
    Vec3 camTarget = camera->pos + camera->forward;
    
    // View matrix
    float fX = camTarget.x - camPos.x;
    float fY = camTarget.y - camPos.y;
    float fZ = camTarget.z - camPos.z;
    float len = sqrt(fX*fX + fY*fY + fZ*fZ);
    fX /= len; fY /= len; fZ /= len;
    
    float upX = camera->up.x, upY = camera->up.y, upZ = camera->up.z;
    float sX = fY * upZ - fZ * upY;
    float sY = fZ * upX - fX * upZ;
    float sZ = fX * upY - fY * upX;
    len = sqrt(sX*sX + sY*sY + sZ*sZ);
    sX /= len; sY /= len; sZ /= len;
    
    float uX = sY * fZ - sZ * fY;
    float uY = sZ * fX - sX * fZ;
    float uZ = sX * fY - sY * fX;
    
    view[0] = sX;  view[4] = sY;  view[8] = sZ;   view[12] = -(sX*camPos.x + sY*camPos.y + sZ*camPos.z);
    view[1] = uX;  view[5] = uY;  view[9] = uZ;   view[13] = -(uX*camPos.x + uY*camPos.y + uZ*camPos.z);
    view[2] = -fX; view[6] = -fY; view[10] = -fZ; view[14] = (fX*camPos.x + fY*camPos.y + fZ*camPos.z);
    view[3] = 0;   view[7] = 0;   view[11] = 0;   view[15] = 1;
    
    // Projection matrix
    float fov = camera->fov * 3.14159f / 180.0f;
    float aspect = (float)width / height;
    float near = 0.1f, far = 100.0f;
    for (int i = 0; i < 16; i++) projection[i] = 0.0f;
    float f = 1.0f / tan(fov * 0.5f);
    projection[0] = f / aspect;
    projection[5] = f;
    projection[10] = (far + near) / (near - far);
    projection[11] = -1.0f;
    projection[14] = (2.0f * far * near) / (near - far);
    
    // Set uniforms
    GLint projLoc = glGetUniformLocation(sphereShader, "projectionMat");
    GLint viewLoc = glGetUniformLocation(sphereShader, "viewMat");
    GLint viewPosLoc = glGetUniformLocation(sphereShader, "viewPos");
    GLint lightPosLoc = glGetUniformLocation(sphereShader, "lightPos");
    GLint pointScaleLoc = glGetUniformLocation(sphereShader, "pointScale");
    
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
    glUniform3f(viewPosLoc, camPos.x, camPos.y, camPos.z);
    glUniform3f(lightPosLoc, lightPosX, lightPosY, lightPosZ);
    glUniform1f(pointScaleLoc, height * projection[5]);
    
    // Update VBO with ball data
    std::vector<float> vertexData;
    vertexData.reserve(balls.size() * 14);
    for (const auto& ball : balls) {
        vertexData.push_back(ball.pos.x);
        vertexData.push_back(ball.pos.y);
        vertexData.push_back(ball.pos.z);
        vertexData.push_back(ball.radius);
        vertexData.push_back(ball.color.x);
        vertexData.push_back(ball.color.y);
        vertexData.push_back(ball.color.z);
        vertexData.push_back(ball.quat.w);
        vertexData.push_back(ball.quat.x);
        vertexData.push_back(ball.quat.y);
        vertexData.push_back(ball.quat.z);
        // Pad to keep alignment (angular velocity not needed in shader)
        vertexData.push_back(0.0f);
        vertexData.push_back(0.0f);
        vertexData.push_back(0.0f);
    }
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_DYNAMIC_DRAW);
    
    // Draw all balls as points
    glDrawArrays(GL_POINTS, 0, balls.size());
    
    glBindVertexArray(0);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_POINT_SPRITE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Copy to CUDA buffer
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    std::vector<unsigned char> pixels(width * height * 4);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    
    for (int y = 0; y < height; y++) {
        cudaMemcpy(d_out + y * width, pixels.data() + (height - 1 - y) * width * 4, width * 4, cudaMemcpyHostToDevice);
    }
    
    glUseProgram(0);
}

void BallsDemo::renderUI() {
    ImGui::Text("Camera Controls:");
    ImGui::Text("  WASD: Move, Q/E: Up/Down");
    ImGui::Text("  Left Mouse: Orbit");
    ImGui::Text("  Middle Mouse: Pan");
    ImGui::Text("  Right Mouse: Rotate View");
    ImGui::Text("  Wheel: Zoom");
    
    ImGui::Separator();
    ImGui::Text("Physics Simulation:");
    ImGui::Text("Number of balls: %d", static_cast<int>(balls.size()));
    
    ImGui::Separator();
    ImGui::Text("Parameters:");
    
    if (ImGui::SliderInt("Ball Count##balls", &numBalls, 1, 200)) {
        initBalls();
    }
    
    ImGui::SliderFloat("Gravity##balls", &gravity, 0.0f, 20.0f, "%.1f");
    ImGui::SliderFloat("Bounce##balls", &bounce, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Friction##balls", &friction, 0.8f, 1.0f, "%.3f");
    ImGui::SliderFloat("Room Size##balls", &roomSize, 5.0f, 20.0f, "%.1f");
    
    ImGui::Separator();
    ImGui::Text("Lighting:");
    ImGui::SliderFloat("Light X##balls", &lightPosX, -10.0f, 10.0f);
    ImGui::SliderFloat("Light Y##balls", &lightPosY, 0.0f, 15.0f);
    ImGui::SliderFloat("Light Z##balls", &lightPosZ, -10.0f, 10.0f);
    
    ImGui::Separator();
    if (ImGui::Button("Reset Simulation##balls", ImVec2(200, 0))) {
        reset();
    }
    
    if (ImGui::Button("Reset View##balls", ImVec2(200, 0))) {
        if (camera) camera->resetView();
    }
}

void BallsDemo::reset() {
    gravity = 9.8f;
    bounce = 0.85f;
    friction = 0.99f;
    roomSize = 5.0f;
    lightPosX = 5.0f;
    lightPosY = 8.0f;
    lightPosZ = 5.0f;
    initBalls();
    if (camera) camera->resetView();
}

