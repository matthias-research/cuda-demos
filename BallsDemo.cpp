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
uniform vec3 lightDir;

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
    coord.y = -coord.y;  // Flip Y axis (gl_PointCoord Y is inverted)
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
    
    // Phong lighting (directional light)
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
    bvhBuilder = new BVHBuilder();
    
    // Initialize mesh renderer
    renderer = new Renderer();
    renderer->init();
    
    // Optionally load a static mesh (uncomment when you have a .glb file)
    staticMesh = new Mesh();
    if (staticMesh->load("assets/cliff.glb")) {
        showMesh = true;
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
    if (staticMesh) {
        staticMesh->cleanup();
        delete staticMesh;
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
        initCudaPhysics(numBalls, roomSize, minRadius, maxRadius, vbo, &cudaVboResource, bvhBuilder);
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
    float near = 0.1f, far = 1000.0f;  // Increased far plane for large meshes
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
    GLint lightDirLoc = glGetUniformLocation(sphereShader, "lightDir");
    GLint pointScaleLoc = glGetUniformLocation(sphereShader, "pointScale");
    
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
    glUniform3f(viewPosLoc, camPos.x, camPos.y, camPos.z);
    
    // Directional light (normalized direction)
    Vec3 lightDirection = Vec3(lightDirX, lightDirY, lightDirZ);
    lightDirection.normalize();
    glUniform3f(lightDirLoc, lightDirection.x, lightDirection.y, lightDirection.z);
    glUniform1f(pointScaleLoc, height * projection[5]);
    
    // Draw all balls as points (VBO is already filled by CUDA)
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, numBalls);
    glBindVertexArray(0);
    
    // Render static mesh after balls (for proper depth testing)
    if (showMesh && staticMesh && renderer) {
        // Sync light direction (normalized, same as balls)
        Vec3 lightDirection = Vec3(lightDirX, lightDirY, lightDirZ);
        lightDirection.normalize();
        renderer->getLight().x = lightDirection.x;
        renderer->getLight().y = lightDirection.y;
        renderer->getLight().z = lightDirection.z;
        
        // Disable backface culling to see both sides
        glDisable(GL_CULL_FACE);
        
        // Vertices are already transformed, so pass identity matrix to renderer
        renderer->renderMesh(*staticMesh, camera, width, height, 1.0f, nullptr);
    }
    
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
    ImGui::SliderFloat("Light Dir X##balls", &lightDirX, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Dir Y##balls", &lightDirY, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Dir Z##balls", &lightDirZ, -1.0f, 1.0f);
    
    ImGui::Separator();
    ImGui::Text("Static Mesh:");
    if (staticMesh) {
        ImGui::Checkbox("Show Mesh##balls", &showMesh);
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
    lightDirX = 0.3f;
    lightDirY = 1.0f;
    lightDirZ = 0.5f;
    
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
    // Mesh rendering is now done inside render() method
    // This method is kept for interface compatibility
}
