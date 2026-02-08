#include "SurfaceRenderer.h"
#include "RenderUtils.h"
#include "Camera.h"
#include <iostream>
#include <cstring>
#include <cmath>

//-----------------------------------------------------------------------------
// Shader sources
//-----------------------------------------------------------------------------

// Vertex shader for point sprites (depth and thickness passes)
static const char* pointSpriteVertexShader = R"(
#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 4) in float aLifetime;  // Lifetime attribute

uniform mat4 viewMat;
uniform mat4 projMat;
uniform float pointScale;
uniform float particleRadius;  // Base particle radius (uniform)

out vec3 eyePos;
out float radius;
out float lifetime;

void main() {
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projMat * eyeSpacePos;
    eyePos = eyeSpacePos.xyz;
    radius = particleRadius;
    lifetime = aLifetime;
    
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = radius * (pointScale / dist);
}
)";

// Depth fragment shader - outputs eye-space Z
static const char* depthFragmentShader = R"(
#version 430 core
uniform float nearClip;
uniform float farClip;

in vec3 eyePos;
in float radius;
in float lifetime;

out float fragDepth;

float projectZ(float z) {
    return farClip * (z + nearClip) / (z * (farClip - nearClip));
}

void main() {
    if (lifetime <= 0.0) discard;  // Discard particles with zero or negative lifetime
    
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    float h = sqrt(1.0 - r2);
    float z = eyePos.z + h * radius;  // Sphere surface Z (right-handed, negative Z into screen)
    
    fragDepth = z;  // Output unprojected Z for smoothing
    gl_FragDepth = projectZ(z) * 0.5 + 0.5;
}
)";

// Thickness fragment shader - additive blending
static const char* thicknessFragmentShader = R"(
#version 430 core
in float lifetime;

out float fragThickness;

void main() {
    if (lifetime <= 0.0) discard;  // Discard particles with zero or negative lifetime
    
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    // Sphere profile gives more thickness in the center
    float h = sqrt(1.0 - r2);
    fragThickness = h;  // Additive blending accumulates thickness
}
)";

// Curvature flow compute shader
static const char* curvatureFlowComputeShader = R"(
#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) uniform image2D depthIn;
layout(r32f, binding = 1) uniform image2D depthOut;

uniform float dt;
uniform float zContrib;
uniform float projW;
uniform float projH;
uniform ivec2 imageSize;

float readDepth(ivec2 pos) {
    if (pos.x < 0 || pos.x >= imageSize.x || pos.y < 0 || pos.y >= imageSize.y) return 0.0;
    return imageLoad(depthIn, pos).r;
}

float diffZ(ivec2 pos, ivec2 offset) {
    float dp = readDepth(pos + offset);
    float dm = readDepth(pos - offset);
    if (dp == 0.0 || dm == 0.0) return 0.0;
    return (dp - dm) * 0.5;
}

float diffZ_2(ivec2 pos, ivec2 offset) {
    float dp = readDepth(pos + offset);
    float d = readDepth(pos);
    float dm = readDepth(pos - offset);
    return dp - 2.0 * d + dm;
}

float diffZ_xy(ivec2 pos) {
    float pp = readDepth(pos + ivec2(1, 1));
    float pm = readDepth(pos + ivec2(1, -1));
    float mp = readDepth(pos + ivec2(-1, 1));
    float mm = readDepth(pos + ivec2(-1, -1));
    return (pp - pm - mp + mm) * 0.25;
}

float computeMeanCurvature(ivec2 pos) {
    float z = readDepth(pos);
    float z_x = diffZ(pos, ivec2(1, 0));
    float z_y = diffZ(pos, ivec2(0, 1));
    
    float Cx = -2.0 / (float(imageSize.x) * projW);
    float Cy = -2.0 / (float(imageSize.y) * projH);
    
    float D = Cy * Cy * z_x * z_x + Cx * Cx * z_y * z_y + Cx * Cx * Cy * Cy * z * z;
    
    // Prevent division by zero for flat surfaces
    const float D_MIN = 1e-12;
    if (D < D_MIN) return 0.0;
    
    float z_xx = diffZ_2(pos, ivec2(1, 0));
    float z_yy = diffZ_2(pos, ivec2(0, 1));
    float z_xy = diffZ_xy(pos);
    
    float D_x = 2.0 * Cy * Cy * z_x * z_xx + 2.0 * Cx * Cx * z_y * z_xy + 2.0 * Cx * Cx * Cy * Cy * z * z_x;
    float D_y = 2.0 * Cy * Cy * z_x * z_xy + 2.0 * Cx * Cx * z_y * z_yy + 2.0 * Cx * Cx * Cy * Cy * z * z_y;
    
    float Ex = 0.5 * z_x * D_x - z_xx * D;
    float Ey = 0.5 * z_y * D_y - z_yy * D;
    
    float sqrtD = sqrt(D);
    float H = (Cy * Ex + Cx * Ey) / (2.0 * D * sqrtD);
    
    // Clamp curvature to prevent extreme corrections
    const float MAX_CURVATURE = 1.0;
    return clamp(H, -MAX_CURVATURE, MAX_CURVATURE);
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if (pos.x >= imageSize.x || pos.y >= imageSize.y) return;
    
    float depth = readDepth(pos);
    
    if (depth == 0.0) {
        imageStore(depthOut, pos, vec4(depth));
        return;
    }
    
    float z_x = diffZ(pos, ivec2(1, 0));
    float z_y = diffZ(pos, ivec2(0, 1));
    
    float meanCurv = computeMeanCurvature(pos);
    
    // Adaptive smoothing: more smoothing where depth varies a lot
    depth += meanCurv * dt * (1.0 + (abs(z_x) + abs(z_y)) * zContrib);
    
    imageStore(depthOut, pos, vec4(depth));
}
)";

// Fullscreen quad vertex shader
static const char* fullscreenVertexShader = R"(
#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 texCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    texCoord = aTexCoord;
}
)";

// Final compositing shader
static const char* compositeFragmentShader = R"(
#version 430 core
uniform sampler2D depthTex;
uniform sampler2D thicknessTex;
uniform mat4 viewMat;
uniform float nearClip;
uniform float farClip;
uniform float projW;
uniform float projH;
uniform vec2 windowSize;
uniform vec3 lightDir;
uniform float radius;  // Particle radius for thickness scaling

// Fluid appearance (from FluidStyle)
uniform vec3 absorptionCoeff;
uniform float absorptionScale;
uniform float specularPower;
uniform float fresnelPower;
uniform float ambientStrength;

in vec2 texCoord;
out vec4 fragColor;

float projectZ(float z) {
    return farClip * (z + nearClip) / (z * (farClip - nearClip));
}

vec3 uvToEye(vec2 uv, float z) {
    uv = uv * 2.0 - 1.0;
    return vec3(-uv.x / projW, -uv.y / projH, 1.0) * z;
}

float diffZ(vec2 uv, vec2 offset) {
    float dp = texture(depthTex, uv + offset).r;
    float dm = texture(depthTex, uv - offset).r;
    if (dm == 0.0) return dp - texture(depthTex, uv).r;
    if (dp == 0.0) return 0.0;
    return (dp - dm) * 0.5;
}

vec3 computeNormal(vec2 uv) {
    float z = texture(depthTex, uv).r;
    vec2 texelSize = 1.0 / windowSize;
    
    float z_x = diffZ(uv, vec2(texelSize.x, 0.0));
    float z_y = diffZ(uv, vec2(0.0, texelSize.y));
    
    float Cx = -2.0 / (windowSize.x * projW);
    float Cy = -2.0 / (windowSize.y * projH);
    
    vec2 screenPos = uv * windowSize;
    float Wx = (windowSize.x - 2.0 * screenPos.x) / (windowSize.x * projW);
    float Wy = (windowSize.y - 2.0 * screenPos.y) / (windowSize.y * projH);
    
    vec3 dx = vec3(Cx * z + Wx * z_x, Wy * z_x, z_x);
    vec3 dy = vec3(Wx * z_y, Cy * z + Wy * z_y, z_y);
    
    vec3 normal = cross(dx, dy);
    float len = length(normal);
    
    // Fall back to view-facing normal for flat surfaces
    if (len < 1e-6) {
        return vec3(0.0, 0.0, 1.0);
    }
    normal = normal / len;
    
    // Transform to world space
    mat3 invViewMat = transpose(mat3(viewMat));
    return invViewMat * normal;
}

void main() {
    float depth = texture(depthTex, texCoord).r;
    if (depth == 0.0) discard;
    
    gl_FragDepth = projectZ(depth) * 0.5 + 0.5;
    
    float thickness = texture(thicknessTex, texCoord).r * radius;
    vec3 normal = computeNormal(texCoord);
    
    // Beer's law absorption
    vec3 absorption = vec3(
        exp(-absorptionCoeff.r * thickness * absorptionScale),
        exp(-absorptionCoeff.g * thickness * absorptionScale),
        exp(-absorptionCoeff.b * thickness * absorptionScale)
    );
    float alpha = 1.0 - exp(-3.0 * thickness * absorptionScale);
    
    // Diffuse lighting
    float diffuse = abs(dot(lightDir, normal)) * 0.5 + 0.5;
    
    // Specular (Fresnel)
    vec3 pos3D = uvToEye(texCoord, depth);
    mat3 invViewMat = transpose(mat3(viewMat));
    pos3D = invViewMat * pos3D;
    
    vec3 eyePos = -vec3(viewMat[3]) * invViewMat;
    vec3 viewDir = normalize(eyePos - pos3D);
    
    float normalReflectance = pow(clamp(dot(normal, lightDir), 0.0, 1.0), specularPower / 10.0);
    float fresnel = normalReflectance + (1.0 - normalReflectance) * pow(1.0 - abs(dot(normal, viewDir)), fresnelPower);
    float specular = clamp(0.1 * thickness, 0.0, 1.0) * fresnel;
    
    vec3 finalColor = (ambientStrength + diffuse * (1.0 - ambientStrength)) * absorption + specular;
    fragColor = clamp(vec4(finalColor, alpha), 0.0, 1.0);
}
)";

//-----------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------

static GLuint compileComputeShader(const char* source) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Compute shader compilation error: " << log << std::endl;
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        std::cerr << "Compute shader link error: " << log << std::endl;
    }
    
    glDeleteShader(shader);
    return program;
}

static void checkGLError(const char* location) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error at " << location << ": 0x" << std::hex << err << std::dec << std::endl;
    }
}

//-----------------------------------------------------------------------------
// SurfaceRenderer implementation
//-----------------------------------------------------------------------------

SurfaceRenderer::SurfaceRenderer() {}

SurfaceRenderer::~SurfaceRenderer() {
    cleanup();
}

bool SurfaceRenderer::init(int maxPts, const PointRenderer::AttribLayout& attribLayout) {
    maxParticles = maxPts;
    layout = attribLayout;
    
    initShaders();
    
    // Create fullscreen quad
    float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
    };
    
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    return true;
}

void SurfaceRenderer::initShaders() {
    depthShader = RenderUtils::createShaderProgram(pointSpriteVertexShader, depthFragmentShader);
    thicknessShader = RenderUtils::createShaderProgram(pointSpriteVertexShader, thicknessFragmentShader);
    curvatureFlowCompute = compileComputeShader(curvatureFlowComputeShader);
    compositeShader = RenderUtils::createShaderProgram(fullscreenVertexShader, compositeFragmentShader);
}

void SurfaceRenderer::initFramebuffers(int width, int height) {
    if (fbWidth == width && fbHeight == height) return;
    
    // Cleanup old resources
    if (depthFBO) glDeleteFramebuffers(1, &depthFBO);
    if (thicknessFBO) glDeleteFramebuffers(1, &thicknessFBO);
    if (depthTexture) glDeleteTextures(1, &depthTexture);
    if (depthTexture2) glDeleteTextures(1, &depthTexture2);
    if (thicknessTexture) glDeleteTextures(1, &thicknessTexture);
    if (depthRBO) glDeleteRenderbuffers(1, &depthRBO);
    
    fbWidth = width;
    fbHeight = height;
    
    // Create depth textures (ping-pong for curvature flow)
    auto createR32FTexture = [&]() -> GLuint {
        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        return tex;
    };
    
    depthTexture = createR32FTexture();
    depthTexture2 = createR32FTexture();
    thicknessTexture = createR32FTexture();
    
    // Create depth renderbuffer for depth testing
    glGenRenderbuffers(1, &depthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    
    // Depth FBO
    glGenFramebuffers(1, &depthFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTexture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Depth FBO not complete!" << std::endl;
    }
    
    // Thickness FBO (no depth testing needed)
    glGenFramebuffers(1, &thicknessFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, thicknessTexture, 0);
    
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Thickness FBO not complete!" << std::endl;
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SurfaceRenderer::setupParticleVAO(GLuint particleVBO) {
    if (currentVBO == particleVBO && particleVAO != 0) return;
    
    currentVBO = particleVBO;
    
    if (particleVAO) glDeleteVertexArrays(1, &particleVAO);
    
    glGenVertexArrays(1, &particleVAO);
    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    
    int stride = layout.strideFloats * sizeof(float);
    
    // Position (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)(layout.posOffset * sizeof(float)));
    glEnableVertexAttribArray(0);
    
    // Radius (location 1)
    if (layout.radiusOffset >= 0) {
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, (void*)(layout.radiusOffset * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    
    glBindVertexArray(0);
}

void SurfaceRenderer::render(Camera* camera, GLuint particleVBO, int particleCount,
                             float particleRadius, const Vec3& lightDir, int width, int height) {
    if (!camera || particleCount <= 0) return;
    
    // Initialize/resize framebuffers if needed
    initFramebuffers(width, height);
    setupParticleVAO(particleVBO);
    
    // Save current OpenGL state
    GLint savedViewport[4];
    glGetIntegerv(GL_VIEWPORT, savedViewport);
    GLboolean savedBlend = glIsEnabled(GL_BLEND);
    GLboolean savedDepthTest = glIsEnabled(GL_DEPTH_TEST);
    
    // Compute actual radii for each pass (matching pysph-cpp)
    float depthRadius = particleRadius * style.particleScale;
    float thicknessRadius = particleRadius * style.thicknessScale;
    
    // Pass 1: Render depth
    renderDepthPass(camera, particleCount, depthRadius, width, height);
    
    // Pass 2: Render thickness
    renderThicknessPass(camera, particleCount, thicknessRadius, width, height);
    
    // Pass 3: Curvature flow smoothing
    if (style.smoothingIterations > 0) {
        runCurvatureFlowCompute(camera, width, height);
    }
    
    // Pass 4: Final composite
    renderComposite(camera, particleRadius, lightDir, width, height);
    
    // Restore OpenGL state
    glViewport(savedViewport[0], savedViewport[1], savedViewport[2], savedViewport[3]);
    if (savedBlend) glEnable(GL_BLEND); else glDisable(GL_BLEND);
    if (savedDepthTest) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
}

void SurfaceRenderer::renderDepthPass(Camera* camera, int particleCount, float radius, int width, int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glViewport(0, 0, fbWidth, fbHeight);
    
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    glUseProgram(depthShader);
    
    glUniformMatrix4fv(glGetUniformLocation(depthShader, "viewMat"), 1, GL_FALSE, camera->viewMat);
    glUniformMatrix4fv(glGetUniformLocation(depthShader, "projMat"), 1, GL_FALSE, camera->projMat);
    glUniform1f(glGetUniformLocation(depthShader, "pointScale"), height * camera->projMat[5]);
    glUniform1f(glGetUniformLocation(depthShader, "particleRadius"), radius);
    glUniform1f(glGetUniformLocation(depthShader, "nearClip"), camera->nearClip);
    glUniform1f(glGetUniformLocation(depthShader, "farClip"), camera->farClip);
    
    glBindVertexArray(particleVAO);
    glDrawArrays(GL_POINTS, 0, particleCount);
    glBindVertexArray(0);
    
    glDisable(GL_POINT_SPRITE);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SurfaceRenderer::renderThicknessPass(Camera* camera, int particleCount, float radius, int width, int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFBO);
    glViewport(0, 0, fbWidth, fbHeight);
    
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Additive blending for thickness accumulation
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    glUseProgram(thicknessShader);
    
    glUniformMatrix4fv(glGetUniformLocation(thicknessShader, "viewMat"), 1, GL_FALSE, camera->viewMat);
    glUniformMatrix4fv(glGetUniformLocation(thicknessShader, "projMat"), 1, GL_FALSE, camera->projMat);
    glUniform1f(glGetUniformLocation(thicknessShader, "pointScale"), height * camera->projMat[5]);
    glUniform1f(glGetUniformLocation(thicknessShader, "particleRadius"), radius);
    
    glBindVertexArray(particleVAO);
    glDrawArrays(GL_POINTS, 0, particleCount);
    glBindVertexArray(0);
    
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_BLEND);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SurfaceRenderer::runCurvatureFlowCompute(Camera* camera, int width, int height) {
    glUseProgram(curvatureFlowCompute);
    
    // Extract projection parameters from projection matrix
    float projW = camera->projMat[0];
    float projH = camera->projMat[5];
    
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "dt"), style.smoothingDt);
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "zContrib"), style.smoothingZContrib);
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "projW"), projW);
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "projH"), projH);
    glUniform2i(glGetUniformLocation(curvatureFlowCompute, "imageSize"), fbWidth, fbHeight);
    
    int numGroupsX = (fbWidth + 15) / 16;
    int numGroupsY = (fbHeight + 15) / 16;
    
    // Ping-pong between depth textures
    for (int i = 0; i < style.smoothingIterations; i++) {
        // Pass 1: depthTexture -> depthTexture2
        glBindImageTexture(0, depthTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, depthTexture2, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glDispatchCompute(numGroupsX, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        
        // Pass 2: depthTexture2 -> depthTexture
        glBindImageTexture(0, depthTexture2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, depthTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glDispatchCompute(numGroupsX, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
    
    glUseProgram(0);
}

void SurfaceRenderer::renderComposite(Camera* camera, float radius, const Vec3& lightDir, int width, int height) {
    // Render to screen with blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    
    glUseProgram(compositeShader);
    
    // Extract projection parameters
    float projW = camera->projMat[0];
    float projH = camera->projMat[5];
    
    glUniform1i(glGetUniformLocation(compositeShader, "depthTex"), 0);
    glUniform1i(glGetUniformLocation(compositeShader, "thicknessTex"), 1);
    glUniformMatrix4fv(glGetUniformLocation(compositeShader, "viewMat"), 1, GL_FALSE, camera->viewMat);
    glUniform1f(glGetUniformLocation(compositeShader, "nearClip"), camera->nearClip);
    glUniform1f(glGetUniformLocation(compositeShader, "farClip"), camera->farClip);
    glUniform1f(glGetUniformLocation(compositeShader, "projW"), projW);
    glUniform1f(glGetUniformLocation(compositeShader, "projH"), projH);
    glUniform2f(glGetUniformLocation(compositeShader, "windowSize"), (float)width, (float)height);
    glUniform1f(glGetUniformLocation(compositeShader, "radius"), radius);
    
    // Transform light direction to view space
    Vec3 normalizedLight = lightDir.normalized();
    float viewSpaceLightX = camera->viewMat[0] * normalizedLight.x + camera->viewMat[4] * normalizedLight.y + camera->viewMat[8] * normalizedLight.z;
    float viewSpaceLightY = camera->viewMat[1] * normalizedLight.x + camera->viewMat[5] * normalizedLight.y + camera->viewMat[9] * normalizedLight.z;
    float viewSpaceLightZ = camera->viewMat[2] * normalizedLight.x + camera->viewMat[6] * normalizedLight.y + camera->viewMat[10] * normalizedLight.z;
    glUniform3f(glGetUniformLocation(compositeShader, "lightDir"), viewSpaceLightX, viewSpaceLightY, viewSpaceLightZ);
    
    // Fluid style parameters
    glUniform3f(glGetUniformLocation(compositeShader, "absorptionCoeff"), 
                style.absorptionCoeff.x, style.absorptionCoeff.y, style.absorptionCoeff.z);
    glUniform1f(glGetUniformLocation(compositeShader, "absorptionScale"), style.absorptionScale);
    glUniform1f(glGetUniformLocation(compositeShader, "specularPower"), style.specularPower);
    glUniform1f(glGetUniformLocation(compositeShader, "fresnelPower"), style.fresnelPower);
    glUniform1f(glGetUniformLocation(compositeShader, "ambientStrength"), style.ambientStrength);
    
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, thicknessTexture);
    
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    glUseProgram(0);
    glDisable(GL_BLEND);
}

void SurfaceRenderer::cleanup() {
    if (particleVAO) { glDeleteVertexArrays(1, &particleVAO); particleVAO = 0; }
    if (quadVAO) { glDeleteVertexArrays(1, &quadVAO); quadVAO = 0; }
    if (quadVBO) { glDeleteBuffers(1, &quadVBO); quadVBO = 0; }
    
    if (depthFBO) { glDeleteFramebuffers(1, &depthFBO); depthFBO = 0; }
    if (thicknessFBO) { glDeleteFramebuffers(1, &thicknessFBO); thicknessFBO = 0; }
    
    if (depthTexture) { glDeleteTextures(1, &depthTexture); depthTexture = 0; }
    if (depthTexture2) { glDeleteTextures(1, &depthTexture2); depthTexture2 = 0; }
    if (thicknessTexture) { glDeleteTextures(1, &thicknessTexture); thicknessTexture = 0; }
    if (depthRBO) { glDeleteRenderbuffers(1, &depthRBO); depthRBO = 0; }
    
    if (depthShader) { glDeleteProgram(depthShader); depthShader = 0; }
    if (thicknessShader) { glDeleteProgram(thicknessShader); thicknessShader = 0; }
    if (curvatureFlowCompute) { glDeleteProgram(curvatureFlowCompute); curvatureFlowCompute = 0; }
    if (compositeShader) { glDeleteProgram(compositeShader); compositeShader = 0; }
    
    currentVBO = 0;
    fbWidth = 0;
    fbHeight = 0;
}
