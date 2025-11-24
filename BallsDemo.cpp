#include "BallsDemo.h"
#include <imgui.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Ball shader using point sprites
const char* ballVertexShader = R"(
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
out vec3 eyePos;
out vec4 quat;
out vec3 viewRight;
out vec3 viewUp;

void main()
{
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    fragPos = aPos;
    radius = aRadius;
    eyePos = eyeSpacePos.xyz;
    quat = aQuat;
    
    // Extract right and up vectors from view matrix (robust for all orientations)
    viewRight = normalize(vec3(viewMat[0][0], viewMat[1][0], viewMat[2][0]));
    
    // Calculate point size based on radius and distance
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = aRadius * (pointScale / dist);
}
)";

const char* ballFragmentShader = R"(
#version 330 core

const float PI = 3.14159265359;

uniform vec3 viewPos;
uniform vec3 lightDir;

in vec3 fragPos;
in float radius;
in vec3 eyePos;
in vec4 quat;
in vec3 viewRight;

out vec4 fragColor;

// Quaternion rotation function
vec3 qtransform(vec4 q, vec3 v)
{
    return v + 2.0 * cross(cross(v, q.xyz) + q.w * v, q.xyz);
}

void main()
{
    // Calculate ball normal from point coordinate
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    coord.y = -coord.y;  // Flip Y axis (gl_PointCoord Y is inverted)
    float r2 = dot(coord, coord);
    
    // Discard fragments outside ball
    if (r2 > 1.0)
        discard;
    
    // Calculate 3D position on ball surface
    float h = sqrt(1.0 - r2);
    
    // Use view matrix axes directly (robust for all camera orientations)
    vec3 axisZ = normalize(fragPos - viewPos);
    vec3 axisX = normalize(viewRight);
    vec3 axisY = cross(axisZ, axisX);
    
    // Calculate surface position using view matrix axes
    vec3 localPos = radius * (coord.x * axisX + coord.y * axisY - h * axisZ);
    vec3 surfacePos = fragPos + localPos;
    
    // Calculate proper world-space normal for lighting
    vec3 normal = normalize(localPos);
    
    // Rotate normal by quaternion
    vec3 rotNormal = qtransform(quat, normal);
    
    // Create classic beach ball pattern with 6 colored segments
    float angle = atan(rotNormal.z, rotNormal.x);
    int segment = int((angle + PI) / (PI * 2.0) * 6.0);
    segment = segment % 6;

    if (abs(rotNormal.y) > 0.95) {
        segment = 5; // Top and bottom segments are white
    }
    
    // Define beach ball colors
    vec3 colors[6];
    colors[0] = vec3(1.0, 0.2, 0.2);   // Red
    colors[1] = vec3(1.0, 0.5, 0.2);  // Orange
    colors[2] = vec3(1.0, 0.95, 0.2); // Yellow
    colors[3] = vec3(0.2, 1.0, 0.2); // Green
    colors[4] = vec3(0.2, 0.2, 1.0);  // Blue  
    colors[5] = vec3(1.0, 1.0, 1.0); // White
    
    vec3 color = colors[segment];
    
    // Phong lighting
    float diffuse = max(0.0, dot(lightDir, normal));
    
    vec3 halfwayDir = normalize(lightDir - axisZ);
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 100.0);
    
    float ambient = 0.4;
    
    vec3 finalColor = color * (ambient + diffuse * 0.6) + vec3(1.0) * specular * 0.5;
    fragColor = vec4(finalColor, 1.0);
}
)";

// Ball shader using point sprites
const char* ballShadowVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aRadius;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform float pointScale;

void main()
{
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    // For orthographic projection, size doesn't change with distance
    gl_PointSize = aRadius * pointScale;
}
)";

const char* ballShadowFragmentShader = R"(
#version 330 core

out vec4 fragColor;

void main()
{
    // Draw simple solid ball for shadow map
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    
    // Discard fragments outside ball
    if (r2 > 1.0)
        discard;
    
    // Output depth as grayscale for visualization
    // gl_FragCoord.z is in [0,1] range (non-linear depth)
    float depth = gl_FragCoord.z;
    fragColor = vec4(depth, depth, depth, 1.0);
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

BallsDemo::BallsDemo() : vao(0), vbo(0), ballShader(0), ballShadowShader(0), 
                         shadowFBO(0), shadowTexture(0), shadowWidth(0), shadowHeight(0) {
    bvhBuilder = new BVHBuilder();
    
    // Initialize mesh renderer
    renderer = new Renderer();
    renderer->init();
    
    // Optionally load a static scene (uncomment when you have a .glb file)
    scene = new Scene();
    if (scene->load("assets/bunny.glb")) {
        showScene = true;
    }
    
    initGL();
    initBalls();
}

BallsDemo::~BallsDemo() {
    if (cudaVboResource) {
        cleanupCudaPhysics(cudaVboResource);
    }
    if (bvhBuilder) {
        delete bvhBuilder;
    }
    if (scene) {
        scene->cleanup();
        delete scene;
    }
    if (renderer) {
        renderer->cleanup();
        delete renderer;
    }
    cleanupGL();
}

void BallsDemo::initBalls() {
    // Initialize CUDA physics with the VBO
    if (useCuda && vbo != 0) {
        initCudaPhysics(numBalls, roomSize, minRadius, maxRadius, minHeight, vbo, &cudaVboResource, bvhBuilder, scene);
    }
}

void BallsDemo::initGL() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // Allocate VBO (will be filled by CUDA)
    glBufferData(GL_ARRAY_BUFFER, numBalls * 14 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
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
    for (int i = 0; i < 2; i++) {

        GLuint vertexShader = i == 0 ? compileShader(GL_VERTEX_SHADER, ballVertexShader) : 
            compileShader(GL_VERTEX_SHADER, ballShadowVertexShader);
        GLuint fragmentShader = i == 0 ? compileShader(GL_FRAGMENT_SHADER, ballFragmentShader) : 
            compileShader(GL_FRAGMENT_SHADER, ballShadowFragmentShader);

        GLuint& shader = i == 0 ? ballShader : ballShadowShader;
        shader = glCreateProgram();
        glAttachShader(shader, vertexShader);
        glAttachShader(shader, fragmentShader);
        glLinkProgram(shader);
        
        GLint success;
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader linking failed:\n" << infoLog << std::endl;
        }
        
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
}

void BallsDemo::cleanupGL() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ballShader) glDeleteProgram(ballShader);
    if (ballShadowShader) glDeleteProgram(ballShadowShader);
    if (shadowFBO) glDeleteFramebuffers(1, &shadowFBO);
    if (shadowTexture) glDeleteTextures(1, &shadowTexture);
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
    if (!camera) return;
    
    initShadowBuffer(width, height);
    
    // Render to shadow FBO from light's perspective (directional light)
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Enable point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    glUseProgram(ballShadowShader);
    
    // Set up matrices from light's perspective
    float view[16], projection[16];
    
    // Normalize light direction (pointing FROM light TO scene)
    Vec3 lightDirection = lightDir;
    lightDirection.normalize();
    
    // Position light far away along the light direction
    Vec3 lightPos = -lightDirection * (roomSize * 2.0f);  // Position light outside the scene
    Vec3 target(0.0f, 0.0f, 0.0f);  // Look at scene center
    
    // Build view matrix from light's perspective
    Vec3 forward = (target - lightPos).normalized();
    Vec3 right = Vec3(0, 1, 0).cross(forward).normalized();
    Vec3 up = forward.cross(right);
    
    view[0] = right.x;    view[4] = right.y;    view[8] = right.z;     view[12] = -right.dot(lightPos);
    view[1] = up.x;       view[5] = up.y;       view[9] = up.z;        view[13] = -up.dot(lightPos);
    view[2] = -forward.x; view[6] = -forward.y; view[10] = -forward.z; view[14] = forward.dot(lightPos);
    view[3] = 0;          view[7] = 0;          view[11] = 0;          view[15] = 1;
    
    // Orthographic projection (parallel rays for directional light)
    float halfSize = roomSize * 0.6f;  // Cover the room area
    float near = 0.1f;
    float far = roomSize * 4.0f;  // Ensure we capture all objects
    
    for (int i = 0; i < 16; i++) projection[i] = 0.0f;
    projection[0] = 1.0f / halfSize;           // Right
    projection[5] = 1.0f / halfSize;           // Top
    projection[10] = -2.0f / (far - near);     // Far/near
    projection[14] = -(far + near) / (far - near);
    projection[15] = 1.0f;
    
    // Set uniforms for shadow shader
    GLint projLoc = glGetUniformLocation(ballShadowShader, "projectionMat");
    GLint viewLoc = glGetUniformLocation(ballShadowShader, "viewMat");
    GLint pointScaleLoc = glGetUniformLocation(ballShadowShader, "pointScale");
    
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
    // For orthographic projection: pointScale converts world-space radius to screen pixels
    // projection[0] = 1/halfSize, so radius in NDC = radius * projection[0]
    // Then multiply by viewport height/2 to get pixels
    glUniform1f(pointScaleLoc, projection[0] * height);
    
    // Draw all balls as flat white discs
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, numBalls);
    glBindVertexArray(0);
    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_POINT_SPRITE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
}

void BallsDemo::update(float deltaTime) {
    // Track FPS
    if (lastUpdateTime > 0.0f) {
        fps = 1.0f / deltaTime;
    }
    lastUpdateTime = deltaTime;
    
    // Update physics on GPU (only if not paused)
    if (!paused && useCuda && cudaVboResource) {
        updateCudaPhysics(deltaTime, Vec3(0, -gravity, 0), friction, bounce, roomSize, cudaVboResource, useBVH);
    }
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
    ImGui::Text("  Ball Count: %d", numBalls);
    
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
    if (ImGui::SliderInt("Ball Count##balls", &numBalls, 1000, 100000)) {
        ballCountChanged = true;
    }
    
    ImGui::SliderFloat("Gravity##balls", &gravity, 0.0f, 20.0f, "%.1f");
    ImGui::SliderFloat("Bounce##balls", &bounce, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Friction##balls", &friction, 0.8f, 1.0f, "%.3f");
    ImGui::SliderFloat("Room Size##balls", &roomSize, 5.0f, 30.0f, "%.1f");
    ImGui::SliderFloat("Min Height##balls", &minHeight, 0.0f, 5.0f, "%.1f");
    
    ImGui::Separator();
    ImGui::Text("Collision Detection Method:");
    if (ImGui::RadioButton("Hash Grid (Fast)##balls", !useBVH)) {
        useBVH = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("BVH (Compare)##balls", useBVH)) {
        useBVH = true;
    }
    ImGui::Text("  %s", useBVH ? "Using BVH tree traversal" : "Using spatial hash grid");
    
    ImGui::Separator();
    ImGui::Text("Lighting (Directional):");
    ImGui::SliderFloat("Light Dir X##balls", &lightDir.x, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Dir Y##balls", &lightDir.y, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Dir Z##balls", &lightDir.z, -1.0f, 1.0f);
    
    ImGui::Separator();
    ImGui::Text("Static Scene:");
    if (scene) {
        ImGui::Text("  Meshes loaded: %zu", scene->getMeshCount());
        ImGui::Checkbox("Show Scene##balls", &showScene);
        if (renderer) {
            ImGui::SliderFloat("Mesh Ambient##balls", &renderer->getMaterial().ambientStrength, 0.0f, 1.0f);
            ImGui::SliderFloat("Mesh Specular##balls", &renderer->getMaterial().specularStrength, 0.0f, 2.0f);
        }
    } else {
        ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "No mesh loaded");
        ImGui::Text("Place a .glb file in assets/ folder");
    }
    
    ImGui::Separator();
    if (ImGui::Button("Reset Simulation##balls", ImVec2(200, 0))) {
        reset();
        ballCountChanged = true;
    }
    
    if (ImGui::Button("Reset View##balls", ImVec2(200, 0))) {
        if (camera) camera->resetView();
    }
    
    // Handle ball count change
    if (ballCountChanged) {
        // Clean up old CUDA resources
        if (cudaVboResource) {
            cleanupCudaPhysics(cudaVboResource);
            cudaVboResource = nullptr;
        }
        
        // Reallocate VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, numBalls * 14 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        // Reinitialize CUDA
        initBalls();
    }
    
    ImGui::Separator();
    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Using CUDA GPU Acceleration!");
    if (useBVH) {
        ImGui::Text("BVH tree for hierarchical collision detection");
    } else {
        ImGui::Text("Spatial hash grid for O(n) collisions");
    }
}

void BallsDemo::reset() {
    gravity = 9.8f;
    bounce = 0.85f;
    friction = 0.99f;
    roomSize = 20.0f;
    minHeight = 1.0f;
    lightDir = -Vec3(0.3f, 1.0f, 0.5f).normalized();
    
    // Reinitialize CUDA physics
    if (cudaVboResource) {
        cleanupCudaPhysics(cudaVboResource);
        cudaVboResource = nullptr;
    }
    initBalls();
    
    if (camera) camera->resetView();
}

void BallsDemo::onKeyPress(unsigned char key) {
    if (key == 'p' || key == 'P') {
        paused = !paused;
    }
}

void BallsDemo::render3D(int width, int height) {
    if (!camera) return;
    
    // Render directly to default framebuffer (screen)
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Enable point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    glUseProgram(ballShader);
    
    // Set up matrices
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
    float near = 0.1f, far = 1000.0f;
    for (int i = 0; i < 16; i++) projection[i] = 0.0f;
    float f = 1.0f / tan(fov * 0.5f);
    projection[0] = f / aspect;
    projection[5] = f;
    projection[10] = (far + near) / (near - far);
    projection[11] = -1.0f;
    projection[14] = (2.0f * far * near) / (near - far);
    
    // Set uniforms
    GLint projLoc = glGetUniformLocation(ballShader, "projectionMat");
    GLint viewLoc = glGetUniformLocation(ballShader, "viewMat");
    GLint viewPosLoc = glGetUniformLocation(ballShader, "viewPos");
    GLint lightDirLoc = glGetUniformLocation(ballShader, "lightDir");
    GLint pointScaleLoc = glGetUniformLocation(ballShader, "pointScale");
    
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
    glUniform3f(viewPosLoc, camPos.x, camPos.y, camPos.z);
    
    // Directional light (normalized direction)
    Vec3 normalizedLightDir = lightDir.normalized();
    glUniform3f(lightDirLoc, normalizedLightDir.x, normalizedLightDir.y, normalizedLightDir.z);
    glUniform1f(pointScaleLoc, height * projection[5]);
    
    // Draw all balls as points (VBO is already filled by CUDA)
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, numBalls);
    glBindVertexArray(0);
    
    // Render static scene after balls (for proper depth testing)
    if (showScene && scene && renderer) {
        // Sync light direction (normalized, same as balls)
        Vec3 lightDirection = lightDir;
        lightDirection.normalize();
        renderer->getLight().x = lightDirection.x;
        renderer->getLight().y = lightDirection.y;
        renderer->getLight().z = lightDirection.z;
        
        // Disable backface culling to see both sides
        glDisable(GL_CULL_FACE);
        
        // Create shadow map struct with light matrix from renderShadows
        Renderer::ShadowMap shadowMapData;
        shadowMapData.texture = shadowTexture;
        shadowMapData.bias = 0.005f;
        
        // Calculate light view-projection matrix (same as in renderShadows)
        Vec3 lightPos = -lightDirection * (roomSize * 2.0f);
        Vec3 target(0.0f, 0.0f, 0.0f);
        Vec3 forward = (target - lightPos).normalized();
        Vec3 right = Vec3(0, 1, 0).cross(forward).normalized();
        Vec3 up = forward.cross(right);
        
        float lightView[16];
        lightView[0] = right.x;    lightView[4] = right.y;    lightView[8] = right.z;     lightView[12] = -right.dot(lightPos);
        lightView[1] = up.x;       lightView[5] = up.y;       lightView[9] = up.z;        lightView[13] = -up.dot(lightPos);
        lightView[2] = -forward.x; lightView[6] = -forward.y; lightView[10] = -forward.z; lightView[14] = forward.dot(lightPos);
        lightView[3] = 0;          lightView[7] = 0;          lightView[11] = 0;          lightView[15] = 1;
        
        float halfSize = roomSize * 0.6f;
        float near = 0.1f;
        float far = roomSize * 4.0f;
        float lightProj[16];
        for (int i = 0; i < 16; i++) lightProj[i] = 0.0f;
        lightProj[0] = 1.0f / halfSize;
        lightProj[5] = 1.0f / halfSize;
        lightProj[10] = -2.0f / (far - near);
        lightProj[14] = -(far + near) / (far - near);
        lightProj[15] = 1.0f;
        
        // Multiply projection * view to get light space matrix (Column-Major order for OpenGL)
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                shadowMapData.lightMatrix[col * 4 + row] = 0.0f;
                for (int k = 0; k < 4; k++) {
                    // Proj(row, k) * View(k, col)
                    // In column-major: Proj[k*4 + row] * View[col*4 + k]
                    shadowMapData.lightMatrix[col * 4 + row] += lightProj[k * 4 + row] * lightView[col * 4 + k];
                }
            }
        }
        
        // Render all meshes in the scene with shadow map
        for (const Mesh* mesh : scene->getMeshes()) {
            // Vertices are already transformed, so pass identity matrix to renderer
            renderer->renderMesh(*mesh, camera, width, height, 1.0f, nullptr, 0.0f, &shadowMapData);
        }
    }
    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_POINT_SPRITE);
    glUseProgram(0);
    
    // Render shadows to texture (no visualization)
    renderShadows(width, height);
}
