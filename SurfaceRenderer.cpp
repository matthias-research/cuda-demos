#include "SurfaceRenderer.h"
#include "RenderUtils.h"
#include "Camera.h"
#include <cstdio>
#include <cmath>

// Depth pass vertex shader - renders particles as point sprites
static const char* depthVertexShader = R"(
#version 430 core
layout (location = 0) in vec3 aPos;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform float pointScale;
uniform float particleRadius;

out float eyeDepth;

void main() {
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    // Store positive eye-space depth (distance from camera)
    eyeDepth = -eyeSpacePos.z;
    
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = particleRadius * (pointScale / dist);
}
)";

// Depth pass fragment shader - outputs linear eye-space depth with spherical correction
static const char* depthFragmentShader = R"(
#version 430 core

in float eyeDepth;

layout (location = 0) out float fragDepth;

uniform float particleRadius;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    // Sphere depth offset - add the z component of the sphere surface
    float h = sqrt(1.0 - r2);
    
    // Output linear eye-space depth (closer to camera = smaller positive value)
    // Subtract because we want the front of the sphere
    float correctedDepth = eyeDepth - h * particleRadius;
    
    fragDepth = correctedDepth;
    
    // Also write to hardware depth buffer for proper depth testing
    // Convert linear depth to NDC depth
    gl_FragDepth = gl_FragCoord.z;
}
)";

// Thickness pass vertex shader
static const char* thicknessVertexShader = R"(
#version 430 core
layout (location = 0) in vec3 aPos;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform float pointScale;
uniform float particleRadius;

void main() {
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = particleRadius * (pointScale / dist);
}
)";

// Thickness pass fragment shader - outputs constant value for additive blending
static const char* thicknessFragmentShader = R"(
#version 430 core

layout (location = 0) out float fragThickness;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    // Output 1.0 - additive blending will accumulate thickness
    fragThickness = 1.0;
}
)";

// Curvature flow compute shader
// Implements mean curvature smoothing with perspective-correct derivatives
static const char* curvatureFlowComputeShader = R"(
#version 430 core
layout (local_size_x = 16, local_size_y = 16) in;

layout (r32f, binding = 0) uniform image2D depthIn;
layout (r32f, binding = 1) uniform image2D depthOut;

uniform float width;
uniform float height;
uniform float projW;  // projection[0][0]
uniform float projH;  // projection[1][1]
uniform float dt;
uniform float varyingZContrib;

#define DEPTH(coords) imageLoad(depthIn, coords).r

// First derivative of depth using central differences
float diffZ(ivec2 coords, ivec2 offset) {
    float dp = DEPTH(coords + offset);
    float dm = DEPTH(coords - offset);
    if (dp == 0.0 || dm == 0.0) return 0.0;
    return (dp - dm) / 2.0;
}

// Second derivative of depth
float diffZ2(ivec2 coords, ivec2 offset) {
    float dp = DEPTH(coords + offset);
    float d = DEPTH(coords);
    float dm = DEPTH(coords - offset);
    if (dp == 0.0 || dm == 0.0) return 0.0;
    return dp - 2.0 * d + dm;
}

// Mixed partial derivative
float diffZxy(ivec2 coords) {
    ivec2 right = ivec2(1, 0);
    ivec2 down = ivec2(0, 1);
    float pp = DEPTH(coords + right + down);
    float pm = DEPTH(coords + right - down);
    float mp = DEPTH(coords - right + down);
    float mm = DEPTH(coords - right - down);
    if (pp == 0.0 || pm == 0.0 || mp == 0.0 || mm == 0.0) return 0.0;
    return (pp - pm - mp + mm) / 4.0;
}

// Compute mean curvature (divergence of normal)
float computeMeanCurvature(ivec2 coords) {
    float z = DEPTH(coords);
    if (z == 0.0) return 0.0;
    
    ivec2 right = ivec2(1, 0);
    ivec2 down = ivec2(0, 1);
    
    float z_x = diffZ(coords, right);
    float z_y = diffZ(coords, down);
    
    float Cx = -2.0 / (width * projW);
    float Cy = -2.0 / (height * projH);
    
    float sx = float(coords.x);
    float sy = float(coords.y);
    float Wx = (width - 2.0 * sx) / (width * projW);
    float Wy = (height - 2.0 * sy) / (height * projH);
    
    float D = Cy * Cy * z_x * z_x + Cx * Cx * z_y * z_y + Cx * Cx * Cy * Cy * z * z;
    
    if (D < 1e-10) return 0.0;
    
    float z_xx = diffZ2(coords, right);
    float z_yy = diffZ2(coords, down);
    float z_xy = diffZxy(coords);
    
    float D_x = 2.0 * Cy * Cy * z_x * z_xx + 2.0 * Cx * Cx * z_y * z_xy + 2.0 * Cx * Cx * Cy * Cy * z * z_x;
    float D_y = 2.0 * Cy * Cy * z_x * z_xy + 2.0 * Cx * Cx * z_y * z_yy + 2.0 * Cx * Cx * Cy * Cy * z * z_y;
    
    float Ex = 0.5 * z_x * D_x - z_xx * D;
    float Ey = 0.5 * z_y * D_y - z_yy * D;
    
    float H = (Cy * Ex + Cx * Ey) / (2.0 * D * sqrt(D));
    
    return H;
}

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imageSize = imageSize(depthIn);
    
    if (coords.x >= imageSize.x || coords.y >= imageSize.y) return;
    
    float depth = DEPTH(coords);
    
    // Skip background pixels
    if (depth == 0.0) {
        imageStore(depthOut, coords, vec4(0.0));
        return;
    }
    
    ivec2 right = ivec2(1, 0);
    ivec2 down = ivec2(0, 1);
    
    float z_x = diffZ(coords, right);
    float z_y = diffZ(coords, down);
    
    float meanCurv = computeMeanCurvature(coords);
    
    // Adaptive timestep based on depth gradient
    // Areas with high depth variation need more smoothing
    float adaptiveDt = dt * (1.0 + (abs(z_y) + abs(z_x)) * varyingZContrib);
    
    depth += meanCurv * adaptiveDt;
    
    imageStore(depthOut, coords, vec4(depth, 0.0, 0.0, 1.0));
}
)";

// Composite vertex shader - fullscreen quad
static const char* compositeVertexShader = R"(
#version 430 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 texCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    texCoord = aTexCoord;
}
)";

// Composite fragment shader - perspective-correct normals + Beer's law + Fresnel
static const char* compositeFragmentShader = R"(
#version 430 core

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D depthTex;
uniform sampler2D thicknessTex;

uniform float width;
uniform float height;
uniform float projW;  // projection[0][0]
uniform float projH;  // projection[1][1]
uniform float nearClip;
uniform float farClip;

uniform vec3 lightDir;
uniform vec3 eyePos;
uniform mat4 invViewMat;

uniform vec3 absorptionCoeff;
uniform float absorptionScale;
uniform float specularPower;
uniform float fresnelPower;
uniform float ambientStrength;
uniform float particleRadius;

// Convert UV + depth to eye-space position
vec3 uvToEye(vec2 uv, float z) {
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(-ndc.x / projW, -ndc.y / projH, 1.0) * z;
}

// Sample depth at texel coordinates
float sampleDepth(ivec2 coords) {
    return texelFetch(depthTex, coords, 0).r;
}

// First derivative of depth
float diffZ(ivec2 coords, ivec2 offset) {
    float dp = sampleDepth(coords + offset);
    float dm = sampleDepth(coords - offset);
    if (dp == 0.0) return sampleDepth(coords) - dm;
    if (dm == 0.0) return dp - sampleDepth(coords);
    return (dp - dm) / 2.0;
}

// Compute perspective-correct normal from depth
vec3 computeNormal(ivec2 coords) {
    float z = sampleDepth(coords);
    if (z == 0.0) return vec3(0.0);
    
    ivec2 right = ivec2(1, 0);
    ivec2 down = ivec2(0, 1);
    
    float z_x = diffZ(coords, right);
    float z_y = diffZ(coords, down);
    
    float Cx = -2.0 / (width * projW);
    float Cy = -2.0 / (height * projH);
    
    float sx = float(coords.x);
    float sy = float(coords.y);
    float Wx = (width - 2.0 * sx) / (width * projW);
    float Wy = (height - 2.0 * sy) / (height * projH);
    
    // Derivative of uvToEye w.r.t. screen x
    vec3 dx = vec3(Cx * z + Wx * z_x, Wy * z_x, z_x);
    
    // Derivative of uvToEye w.r.t. screen y
    vec3 dy = vec3(Wx * z_y, Cy * z + Wy * z_y, z_y);
    
    vec3 normal = normalize(cross(dx, dy));
    
    return normal;
}

// Convert linear depth to NDC depth for depth buffer
float linearToNDC(float linearDepth) {
    // For right-handed system with reversed depth
    return (farClip * (linearDepth - nearClip)) / (linearDepth * (farClip - nearClip));
}

void main() {
    ivec2 coords = ivec2(gl_FragCoord.xy);
    
    float depth = sampleDepth(coords);
    
    // Discard background
    if (depth == 0.0) {
        discard;
    }
    
    // Write proper depth to depth buffer
    gl_FragDepth = linearToNDC(depth);
    
    // Get thickness
    float thickness = texture(thicknessTex, texCoord).r * particleRadius * absorptionScale;
    
    // Compute normal in eye space
    vec3 normal = computeNormal(coords);
    if (length(normal) < 0.5) {
        discard;
    }
    
    // Transform normal to world space
    vec3 worldNormal = normalize(mat3(invViewMat) * normal);
    
    // Eye-space position
    vec3 posEye = uvToEye(texCoord, depth);
    vec3 posWorld = (invViewMat * vec4(posEye, 1.0)).xyz;
    
    // View direction
    vec3 viewDir = normalize(eyePos - posWorld);
    
    // Diffuse lighting
    float NdotL = max(0.0, dot(worldNormal, lightDir));
    float diffuse = NdotL * 0.5 + 0.5;  // Half-lambert for softer look
    
    // Specular (Blinn-Phong)
    vec3 halfVec = normalize(lightDir + viewDir);
    float NdotH = max(0.0, dot(worldNormal, halfVec));
    float specular = pow(NdotH, specularPower);
    
    // Fresnel (Schlick approximation)
    float NdotV = max(0.0, dot(worldNormal, viewDir));
    float fresnel = pow(1.0 - NdotV, fresnelPower);
    
    // Beer's law absorption - different rates for RGB
    vec3 absorption = vec3(
        exp(-absorptionCoeff.x * thickness),
        exp(-absorptionCoeff.y * thickness),
        exp(-absorptionCoeff.z * thickness)
    );
    
    // Alpha based on thickness
    float alpha = 1.0 - exp(-3.0 * thickness);
    
    // Combine lighting
    vec3 baseColor = vec3(0.8, 0.9, 1.0);  // Light blue-white base
    vec3 color = baseColor * absorption * (ambientStrength + diffuse * (1.0 - ambientStrength));
    color += vec3(1.0) * specular * 0.5;
    color += vec3(0.9, 0.95, 1.0) * fresnel * 0.3;
    
    fragColor = vec4(color, alpha);
}
)";

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
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
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
    depthShader = RenderUtils::createShaderProgram(depthVertexShader, depthFragmentShader);
    thicknessShader = RenderUtils::createShaderProgram(thicknessVertexShader, thicknessFragmentShader);
    compositeShader = RenderUtils::createShaderProgram(compositeVertexShader, compositeFragmentShader);
    
    // Compile curvature flow compute shader
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &curvatureFlowComputeShader, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
        fprintf(stderr, "Curvature flow compute shader compilation failed:\n%s\n", infoLog);
    }
    
    curvatureFlowCompute = glCreateProgram();
    glAttachShader(curvatureFlowCompute, shader);
    glLinkProgram(curvatureFlowCompute);
    
    glGetProgramiv(curvatureFlowCompute, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[1024];
        glGetProgramInfoLog(curvatureFlowCompute, 1024, nullptr, infoLog);
        fprintf(stderr, "Curvature flow compute shader linking failed:\n%s\n", infoLog);
    }
    
    glDeleteShader(shader);
}

void SurfaceRenderer::initFramebuffers(int width, int height) {
    if (fbWidth == width && fbHeight == height && depthFBO != 0) {
        return;
    }
    
    // Cleanup old resources
    if (depthFBO != 0) {
        glDeleteFramebuffers(1, &depthFBO);
        glDeleteFramebuffers(1, &thicknessFBO);
        glDeleteTextures(1, &depthTexture);
        glDeleteTextures(1, &depthTexture2);
        glDeleteTextures(1, &thicknessTexture);
        glDeleteRenderbuffers(1, &depthRBO);
    }
    
    fbWidth = width;
    fbHeight = height;
    
    // Depth texture (ping)
    glGenTextures(1, &depthTexture);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Depth texture (pong)
    glGenTextures(1, &depthTexture2);
    glBindTexture(GL_TEXTURE_2D, depthTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Thickness texture
    glGenTextures(1, &thicknessTexture);
    glBindTexture(GL_TEXTURE_2D, thicknessTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Depth renderbuffer
    glGenRenderbuffers(1, &depthRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
    
    // Depth FBO
    glGenFramebuffers(1, &depthFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthTexture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRBO);
    
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "SurfaceRenderer: Depth FBO incomplete: 0x%x\n", status);
    }
    
    // Thickness FBO
    glGenFramebuffers(1, &thicknessFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, thicknessTexture, 0);
    
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "SurfaceRenderer: Thickness FBO incomplete: 0x%x\n", status);
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SurfaceRenderer::setupParticleVAO(GLuint particleVBO) {
    if (currentVBO == particleVBO && particleVAO != 0) {
        return;
    }
    
    if (particleVAO != 0) {
        glDeleteVertexArrays(1, &particleVAO);
    }
    
    currentVBO = particleVBO;
    
    glGenVertexArrays(1, &particleVAO);
    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    
    int stride = layout.strideFloats * sizeof(float);
    
    // Position only
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)(layout.posOffset * sizeof(float)));
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void SurfaceRenderer::render(Camera* camera, GLuint particleVBO, int particleCount,
                              const Vec3& lightDir, int width, int height) {
    if (!camera || particleCount <= 0) return;
    
    initFramebuffers(width, height);
    setupParticleVAO(particleVBO);
    
    // Pass 1: Render particles to depth texture
    renderDepthPass(camera, particleCount, width, height);
    
    // Pass 2: Render particles to thickness texture (additive)
    renderThicknessPass(camera, particleCount, width, height);
    
    // Pass 3: Curvature flow smoothing
    runCurvatureFlowCompute(camera, width, height);
    
    // Pass 4: Final composite
    renderComposite(camera, lightDir, width, height);
}

void SurfaceRenderer::renderDepthPass(Camera* camera, int particleCount, int width, int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, depthFBO);
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    
    glUseProgram(depthShader);
    
    // Build matrices
    float view[16], projection[16];
    Vec3 camPos = camera->pos;
    Vec3 camTarget = camera->pos + camera->forward;
    
    RenderUtils::buildViewMatrix(view, camPos.x, camPos.y, camPos.z,
                                 camTarget.x, camTarget.y, camTarget.z,
                                 camera->up.x, camera->up.y, camera->up.z);
    
    float fov = camera->fov * 3.14159f / 180.0f;
    float aspect = (float)width / height;
    RenderUtils::buildProjectionMatrix(projection, fov, aspect, camera->nearClip, camera->farClip);
    
    glUniformMatrix4fv(glGetUniformLocation(depthShader, "projectionMat"), 1, GL_FALSE, projection);
    glUniformMatrix4fv(glGetUniformLocation(depthShader, "viewMat"), 1, GL_FALSE, view);
    glUniform1f(glGetUniformLocation(depthShader, "pointScale"), height * projection[5]);
    glUniform1f(glGetUniformLocation(depthShader, "particleRadius"), style.particleScale);
    
    glBindVertexArray(particleVAO);
    glDrawArrays(GL_POINTS, 0, particleCount);
    glBindVertexArray(0);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SurfaceRenderer::renderThicknessPass(Camera* camera, int particleCount, int width, int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, thicknessFBO);
    glViewport(0, 0, width, height);
    
    // No depth test - we want to accumulate all particles
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Additive blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    
    glUseProgram(thicknessShader);
    
    // Build matrices
    float view[16], projection[16];
    Vec3 camPos = camera->pos;
    Vec3 camTarget = camera->pos + camera->forward;
    
    RenderUtils::buildViewMatrix(view, camPos.x, camPos.y, camPos.z,
                                 camTarget.x, camTarget.y, camTarget.z,
                                 camera->up.x, camera->up.y, camera->up.z);
    
    float fov = camera->fov * 3.14159f / 180.0f;
    float aspect = (float)width / height;
    RenderUtils::buildProjectionMatrix(projection, fov, aspect, camera->nearClip, camera->farClip);
    
    glUniformMatrix4fv(glGetUniformLocation(thicknessShader, "projectionMat"), 1, GL_FALSE, projection);
    glUniformMatrix4fv(glGetUniformLocation(thicknessShader, "viewMat"), 1, GL_FALSE, view);
    glUniform1f(glGetUniformLocation(thicknessShader, "pointScale"), height * projection[5]);
    glUniform1f(glGetUniformLocation(thicknessShader, "particleRadius"), style.thicknessScale);
    
    glBindVertexArray(particleVAO);
    glDrawArrays(GL_POINTS, 0, particleCount);
    glBindVertexArray(0);
    
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SurfaceRenderer::runCurvatureFlowCompute(Camera* camera, int width, int height) {
    glUseProgram(curvatureFlowCompute);
    
    // Projection matrix values
    float fov = camera->fov * 3.14159f / 180.0f;
    float aspect = (float)width / height;
    float projW = 1.0f / (aspect * tanf(fov * 0.5f));  // projection[0][0]
    float projH = 1.0f / tanf(fov * 0.5f);             // projection[1][1]
    
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "width"), (float)width);
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "height"), (float)height);
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "projW"), projW);
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "projH"), projH);
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "dt"), style.smoothingDt);
    glUniform1f(glGetUniformLocation(curvatureFlowCompute, "varyingZContrib"), style.smoothingZContrib);
    
    int numGroupsX = (width + 15) / 16;
    int numGroupsY = (height + 15) / 16;
    
    // Ping-pong iterations
    for (int i = 0; i < style.smoothingIterations; i++) {
        // Ping: depthTexture -> depthTexture2
        glBindImageTexture(0, depthTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, depthTexture2, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glDispatchCompute(numGroupsX, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        
        // Pong: depthTexture2 -> depthTexture
        glBindImageTexture(0, depthTexture2, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F);
        glBindImageTexture(1, depthTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
        glDispatchCompute(numGroupsX, numGroupsY, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
}

void SurfaceRenderer::renderComposite(Camera* camera, const Vec3& lightDir, int width, int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    
    // Alpha blending for transparent fluid
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glUseProgram(compositeShader);
    
    // Bind textures
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glUniform1i(glGetUniformLocation(compositeShader, "depthTex"), 0);
    
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, thicknessTexture);
    glUniform1i(glGetUniformLocation(compositeShader, "thicknessTex"), 1);
    
    // Projection parameters
    float fov = camera->fov * 3.14159f / 180.0f;
    float aspect = (float)width / height;
    float projW = 1.0f / (aspect * tanf(fov * 0.5f));
    float projH = 1.0f / tanf(fov * 0.5f);
    
    glUniform1f(glGetUniformLocation(compositeShader, "width"), (float)width);
    glUniform1f(glGetUniformLocation(compositeShader, "height"), (float)height);
    glUniform1f(glGetUniformLocation(compositeShader, "projW"), projW);
    glUniform1f(glGetUniformLocation(compositeShader, "projH"), projH);
    glUniform1f(glGetUniformLocation(compositeShader, "nearClip"), camera->nearClip);
    glUniform1f(glGetUniformLocation(compositeShader, "farClip"), camera->farClip);
    
    // Inverse view matrix for world-space calculations
    float view[16], invView[16];
    Vec3 camPos = camera->pos;
    Vec3 camTarget = camera->pos + camera->forward;
    RenderUtils::buildViewMatrix(view, camPos.x, camPos.y, camPos.z,
                                 camTarget.x, camTarget.y, camTarget.z,
                                 camera->up.x, camera->up.y, camera->up.z);
    
    // Simple inverse for view matrix (transpose of rotation, negate translation)
    invView[0] = view[0]; invView[1] = view[4]; invView[2] = view[8];  invView[3] = 0;
    invView[4] = view[1]; invView[5] = view[5]; invView[6] = view[9];  invView[7] = 0;
    invView[8] = view[2]; invView[9] = view[6]; invView[10] = view[10]; invView[11] = 0;
    invView[12] = camPos.x; invView[13] = camPos.y; invView[14] = camPos.z; invView[15] = 1;
    
    glUniformMatrix4fv(glGetUniformLocation(compositeShader, "invViewMat"), 1, GL_FALSE, invView);
    
    // Light and eye position
    Vec3 normalizedLight = lightDir.normalized();
    glUniform3f(glGetUniformLocation(compositeShader, "lightDir"),
                normalizedLight.x, normalizedLight.y, normalizedLight.z);
    glUniform3f(glGetUniformLocation(compositeShader, "eyePos"), camPos.x, camPos.y, camPos.z);
    
    // Style parameters
    glUniform3f(glGetUniformLocation(compositeShader, "absorptionCoeff"),
                style.absorptionCoeff.x, style.absorptionCoeff.y, style.absorptionCoeff.z);
    glUniform1f(glGetUniformLocation(compositeShader, "absorptionScale"), style.absorptionScale);
    glUniform1f(glGetUniformLocation(compositeShader, "specularPower"), style.specularPower);
    glUniform1f(glGetUniformLocation(compositeShader, "fresnelPower"), style.fresnelPower);
    glUniform1f(glGetUniformLocation(compositeShader, "ambientStrength"), style.ambientStrength);
    glUniform1f(glGetUniformLocation(compositeShader, "particleRadius"), style.particleScale);
    
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    
    glDisable(GL_BLEND);
    glUseProgram(0);
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
