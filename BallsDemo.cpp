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
layout (location = 4) in float aShadowValue;

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
flat out float shadowValue;

void main()
{
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    fragPos = aPos;
    radius = aRadius;
    eyePos = eyeSpacePos.xyz;
    quat = aQuat;
    shadowValue = aShadowValue;
    
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
const float MaxFloat = 3.402823466e+38;

uniform vec3 viewPos;
uniform vec3 lightDir;
uniform bool useTexture;
uniform sampler2D ballTexture;

in vec3 fragPos;
in float radius;
in vec3 eyePos;
in vec4 quat;
in vec3 viewRight;
flat in float shadowValue;

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
    vec3 axisY = cross(axisX, axisZ);
    
    // Calculate surface position using view matrix axes
    vec3 localPos = radius * (coord.x * axisX + coord.y * axisY - h * axisZ);
    vec3 surfacePos = fragPos + localPos;
    
    // Calculate proper world-space normal for lighting
    vec3 normal = normalize(localPos);
    
    // Rotate normal by quaternion
    vec3 rotNormal = qtransform(quat, vec3(normal.x, -normal.y, normal.z));
    
    vec3 color;
    
    if (useTexture) {
        float phi = atan(rotNormal.z, rotNormal.x); 
        float u = phi / (2.0 * PI) + 0.5;
        float v = acos(rotNormal.y) / PI;        
        color = texture(ballTexture, vec2(u, v)).rgb;
    } 
    else {
        // Use default beach ball pattern
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
        
        color = colors[segment];
    }
    
    // Phong lighting
    float diffuse = max(0.0, dot(lightDir, normal));
    
    vec3 halfwayDir = normalize(lightDir - axisZ);
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 100.0);
    
    float ambient = 0.4;
    
    vec3 finalColor = color * (ambient + diffuse * 0.6) + vec3(1.0) * specular * 0.5;
    
    // Apply shadow darkening if ball is in shadow
    finalColor *= (1.0 - shadowValue * 0.5);
    
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

BallsDemo::BallsDemo(const BallsDemoDescriptor& desc) : vao(0), vbo(0), ballShader(0), ballShadowShader(0), 
                         shadowFBO(0), shadowTexture(0), shadowWidth(0), shadowHeight(0) {
    demoDesc = desc;
    bvhBuilder = new BVHBuilder();
    
    // Initialize mesh renderer
    renderer = new Renderer();
    renderer->init();
    
    // Scene will be loaded on demand (lazy loading)
    scene = new Scene();
    
    // Initialize skybox
    skybox = new Skybox();
    if (!skybox->loadFromBMP("assets/skybox.bmp")) {
        delete skybox;
        skybox = nullptr;
        showSkybox = false;
    }
    
    // Allocate GPU device data structure using factory
    deviceData = createBallsDeviceData();
    
    paused = true;  // Start in paused mode
    
    initGL();
    initBalls();
    
    // Load ball texture from descriptor if specified
    printf("BallsDemo constructor: ballTextureName = '%s'\n", demoDesc.ballTextureName.c_str());
    if (!demoDesc.ballTextureName.empty()) {
        printf("Loading ball texture from descriptor\n");
        ballTexture = loadTexture("assets/" + demoDesc.ballTextureName);
        useTextureMode = true;  // Automatically enable texture mode when texture is loaded
        printf("Texture mode enabled automatically\n");
    } else {
        printf("No ball texture specified, creating default\n");
        ballTexture = loadTexture("");  // Creates default checkerboard
    }
    printf("Ball texture ID: %u\n", ballTexture);
}

void BallsDemo::ensureSceneLoaded() {
    // Load scene on first use
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
    
    // Check if ball count changed (reinitialize if needed)
    // lastInitializedBallCount starts at -1, so first call will always reinit
    if (lastInitializedBallCount != demoDesc.numBalls) {
        printf("Ball count mismatch (last init: %d, current: %d), reinitializing CUDA physics...\n", lastInitializedBallCount, demoDesc.numBalls);
        if (cudaVboResource) {
            cleanupCudaPhysics(cudaVboResource);
        }
        initCudaPhysics(vbo, &cudaVboResource, bvhBuilder, scene);
        lastInitializedBallCount = demoDesc.numBalls;
    }
}

BallsDemo::~BallsDemo() {
    if (cudaVboResource) {
        cleanupCudaPhysics(cudaVboResource);
    }
    if (deviceData) {
        deleteBallsDeviceData(deviceData);
        deviceData = nullptr;
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
    if (skybox) {
        skybox->cleanup();
        delete skybox;
    }
    cleanupGL();
}

void BallsDemo::initBalls() {
    // Initialize CUDA physics with the VBO
    if (vbo != 0) {
        initCudaPhysics(vbo, &cudaVboResource, bvhBuilder, scene);
    }
}

void BallsDemo::initGL() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // Allocate VBO (will be filled by CUDA)
    glBufferData(GL_ARRAY_BUFFER, demoDesc.numBalls * 14 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
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
    
    // Shadow depth attribute (location 4)
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(11 * sizeof(float)));
    glEnableVertexAttribArray(4);
    
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
    if (ballTexture) glDeleteTextures(1, &ballTexture);
}

GLuint BallsDemo::loadTexture(const std::string& filename) {
    printf("=== loadTexture() called ===\n");
    printf("  filename: '%s'\n", filename.c_str());
    
    // Try to load from BMP file first
    if (!filename.empty()) {
        printf("  Attempting to load BMP from: %s\n", filename.c_str());
        
        FILE* file;
        fopen_s(&file, filename.c_str(), "rb");
        if (file) {
            printf("  File opened successfully\n");
            
            // Read BMP header (54 bytes)
            unsigned char header[54];
            if (fread(header, 1, 54, file) == 54) {
                // Check BMP signature
                if (header[0] == 'B' && header[1] == 'M') {
                    printf("  Valid BMP signature found\n");
                    
                    // Read image info from header
                    unsigned int dataOffset = *(int*)&(header[0x0A]);
                    unsigned int width = *(int*)&(header[0x12]);
                    unsigned int height = *(int*)&(header[0x16]);
                    unsigned short bitsPerPixel = *(short*)&(header[0x1C]);
                    
                    printf("  BMP dimensions: %u x %u, %u bits per pixel\n", width, height, bitsPerPixel);
                    
                    // Support 24-bit and 32-bit
                    if ((bitsPerPixel == 24 || bitsPerPixel == 32) && width > 0 && height > 0) {
                        int bytesPerPixel = bitsPerPixel / 8;
                        int rowSize = width * 3;  // Output is always RGB
                        int paddedRowSize = (width * bytesPerPixel + 3) & ~3;
                        
                        unsigned char* rawData = new unsigned char[paddedRowSize * height];
                        unsigned char* imageData = new unsigned char[rowSize * height];
                        
                        // Read pixel data
                        fseek(file, dataOffset, SEEK_SET);
                        if (fread(rawData, 1, paddedRowSize * height, file) == paddedRowSize * height) {
                            printf("  Pixel data read successfully\n");
                            
                            // Convert BMP (BGR) to RGB and flip vertically
                            for (int y = 0; y < (int)height; y++) {
                                for (int x = 0; x < (int)width; x++) {
                                    int srcIdx = y * paddedRowSize + x * bytesPerPixel;
                                    int dstIdx = (height - 1 - y) * rowSize + x * 3;
                                    imageData[dstIdx + 0] = rawData[srcIdx + 2]; // R (from B)
                                    imageData[dstIdx + 1] = rawData[srcIdx + 1]; // G
                                    imageData[dstIdx + 2] = rawData[srcIdx + 0]; // B (from R)
                                }
                            }
                            
                            // Create OpenGL texture
                            GLuint texture;
                            glGenTextures(1, &texture);
                            printf("  Generated OpenGL texture ID: %u\n", texture);
                            
                            glBindTexture(GL_TEXTURE_2D, texture);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                            
                            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, imageData);
                            glGenerateMipmap(GL_TEXTURE_2D);
                            glBindTexture(GL_TEXTURE_2D, 0);
                            
                            printf("  BMP texture loaded successfully! (ID: %u)\n", texture);
                            
                            delete[] rawData;
                            delete[] imageData;
                            fclose(file);
                            
                            printf("=== loadTexture() returning texture ID: %u ===\n", texture);
                            return texture;
                        }
                        
                        delete[] rawData;
                        delete[] imageData;
                    } else {
                        printf("  ERROR: Unsupported BMP format (expected 24 or 32 bits per pixel)\n");
                    }
                } else {
                    printf("  ERROR: Invalid BMP signature\n");
                }
            } else {
                printf("  ERROR: Failed to read BMP header\n");
            }
            fclose(file);
        } else {
            printf("  ERROR: Could not open file\n");
        }
        printf("  Falling back to procedural checkerboard\n");
    } else {
        printf("  No filename provided, creating procedural checkerboard\n");
    }
    
    // Fallback: Create a simple checkerboard pattern
    const int texSize = 512;
    unsigned char* data = new unsigned char[texSize * texSize * 3];
    printf("  Allocated checkerboard texture data: %d x %d pixels\n", texSize, texSize);
    
    for (int y = 0; y < texSize; y++) {
        for (int x = 0; x < texSize; x++) {
            int idx = (y * texSize + x) * 3;
            int checkerSize = 32;
            bool check = ((x / checkerSize) + (y / checkerSize)) % 2 == 0;
            data[idx + 0] = check ? 255 : 64;      // R
            data[idx + 1] = check ? 128 : 32;      // G
            data[idx + 2] = check ? 200 : 96;      // B
        }
    }
    
    GLuint texture;
    glGenTextures(1, &texture);
    printf("  Generated OpenGL texture ID for checkerboard: %u\n", texture);
    
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texSize, texSize, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    delete[] data;
    printf("=== loadTexture() returning checkerboard texture ID: %u ===\n", texture);
    return texture;
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
    float roomDiag = (demoDesc.sceneBounds.maximum - demoDesc.sceneBounds.minimum).magnitude();
    Vec3 lightPos = -lightDirection * (roomDiag * 2.0f);  // Position light outside the scene
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
    float halfSize = roomDiag * 0.6f;  // Cover the room area
    float near = 0.1f;
    float far = roomDiag * 4.0f;  // Ensure we capture all objects
    
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
    glDrawArrays(GL_POINTS, 0, demoDesc.numBalls);
    glBindVertexArray(0);
    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_POINT_SPRITE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
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
        updateCudaPhysics(deltaTime, cudaVboResource, useBVH);
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
        if (renderer) {
            ImGui::Checkbox("Use Baked Lighting##balls", &demoDesc.useBakedLighting);
            if (!demoDesc.useBakedLighting) {
                ImGui::SliderFloat("Mesh Ambient##balls", &demoDesc.meshAmbient, 0.0f, 1.0f);
                renderer->getMaterial().ambientStrength = demoDesc.meshAmbient;
                ImGui::SliderFloat("Mesh Specular##balls", &renderer->getMaterial().specularStrength, 0.0f, 2.0f);
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
        // Clean up old CUDA resources
        if (cudaVboResource) {
            cleanupCudaPhysics(cudaVboResource);
            cudaVboResource = nullptr;
        }
        
        // Reallocate VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, demoDesc.numBalls * 14 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        
        // Reinitialize CUDA
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
    
    ImGui::Separator();
    if (useBVH) {
        ImGui::Text("BVH tree for hierarchical collision detection");
    } else {
        ImGui::Text("Spatial hash grid for O(n) collisions");
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
    if (!camera) return;
    
    // Apply camera clipping planes
    camera->nearClip = demoDesc.cameraNear;
    camera->farClip = demoDesc.cameraFar;
    
    // Render directly to default framebuffer (screen)
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Render skybox first (always in background)
    if (skybox && showSkybox) {
        skybox->render(camera);
    }
    
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
    float near = camera->nearClip, far = camera->farClip;
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
    GLint useTextureLoc = glGetUniformLocation(ballShader, "useTexture");
    GLint ballTextureLoc = glGetUniformLocation(ballShader, "ballTexture");
    
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
    glUniform3f(viewPosLoc, camPos.x, camPos.y, camPos.z);
    glUniform1i(useTextureLoc, useTextureMode ? 1 : 0);
    
    if (useTextureMode && ballTexture == 0) {
        printf("WARNING: useTextureMode is true but ballTexture is 0\n");
    }
    
    // Bind ball texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, ballTexture);
    glUniform1i(ballTextureLoc, 0);
    
    // Transform light direction to view space so it stays constant relative to camera
    // Extract rotation part of view matrix (upper 3x3) and apply inverse (transpose for rotation)
    Vec3 normalizedLightDir = lightDir.normalized();
    Vec3 viewSpaceLightDir(
        view[0] * normalizedLightDir.x + view[1] * normalizedLightDir.y + view[2] * normalizedLightDir.z,
        view[4] * normalizedLightDir.x + view[5] * normalizedLightDir.y + view[6] * normalizedLightDir.z,
        view[8] * normalizedLightDir.x + view[9] * normalizedLightDir.y + view[10] * normalizedLightDir.z
    );
    glUniform3f(lightDirLoc, viewSpaceLightDir.x, viewSpaceLightDir.y, viewSpaceLightDir.z);
    glUniform1f(pointScaleLoc, height * projection[5]);
    
    // Draw all balls as points (VBO is already filled by CUDA)
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, demoDesc.numBalls);
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
        
        // Set baked lighting mode
        renderer->setUseBakedLighting(demoDesc.useBakedLighting);
        
        // Create shadow map struct with light matrix from renderShadows
        Renderer::ShadowMap shadowMapData;
        shadowMapData.texture = shadowTexture;
        shadowMapData.bias = 0.005f;
        
        // Calculate light view-projection matrix (same as in renderShadows)
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
        float near = 0.1f;
        float far = roomDiag * 4.0f;
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
            renderer->renderMesh(*mesh, camera, width, height, 1.0f, nullptr, 0.0f, useShadows ? &shadowMapData : nullptr);
        }
    }
    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_POINT_SPRITE);
    glUseProgram(0);
    
    // Render shadows to texture (only if enabled)
    if (useShadows) {
        renderShadows(width, height);
    }
}
