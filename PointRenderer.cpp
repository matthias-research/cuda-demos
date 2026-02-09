#include "PointRenderer.h"
#include "RenderUtils.h"
#include "Camera.h"
#include <cmath>

// Basic point shader - solid colored spheres with Phong lighting
static const char* basicVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aRadius;
layout (location = 2) in vec3 aColor;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform float pointScale;

out vec3 fragColor;
out vec3 eyePos;
out float radius;

void main() {
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    fragColor = aColor;
    eyePos = eyeSpacePos.xyz;
    radius = aRadius;
    
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = aRadius * (pointScale / dist);
}
)";

static const char* basicFragmentShader = R"(
#version 330 core
uniform vec3 lightDir;

in vec3 fragColor;
in vec3 eyePos;
in float radius;

out vec4 outColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    coord.y = -coord.y;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    float h = sqrt(1.0 - r2);
    vec3 normal = normalize(vec3(coord.x, coord.y, h));
    
    float diffuse = max(0.0, dot(lightDir, normal));
    float ambient = 0.4;
    
    vec3 halfwayDir = normalize(lightDir + vec3(0.0, 0.0, 1.0));
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    
    vec3 result = fragColor * (ambient + diffuse * 0.6) + vec3(1.0) * specular * 0.3;
    outColor = vec4(result, 1.0);
}
)";

// Textured point shader - spheres with texture mapping (no rotation)
static const char* texturedVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aRadius;
layout (location = 2) in vec3 aColor;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform float pointScale;

out vec3 fragPos;
out vec3 eyePos;
out float radius;
out vec3 viewRight;

void main() {
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    fragPos = aPos;
    eyePos = eyeSpacePos.xyz;
    radius = aRadius;
    viewRight = normalize(vec3(viewMat[0][0], viewMat[1][0], viewMat[2][0]));
    
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = aRadius * (pointScale / dist);
}
)";

static const char* texturedFragmentShader = R"(
#version 330 core
const float PI = 3.14159265359;

uniform vec3 viewPos;
uniform vec3 lightDir;
uniform sampler2D pointTexture;

in vec3 fragPos;
in vec3 eyePos;
in float radius;
in vec3 viewRight;

out vec4 outColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    coord.y = -coord.y;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    float h = sqrt(1.0 - r2);
    
    vec3 axisZ = normalize(fragPos - viewPos);
    vec3 axisX = normalize(viewRight);
    vec3 axisY = cross(axisX, axisZ);
    
    vec3 localPos = radius * (coord.x * axisX + coord.y * axisY - h * axisZ);
    vec3 normal = normalize(localPos);
    
    // Spherical UV mapping
    float phi = atan(normal.z, normal.x);
    float u = phi / (2.0 * PI) + 0.5;
    float v = acos(clamp(normal.y, -1.0, 1.0)) / PI;
    vec3 color = texture(pointTexture, vec2(u, v)).rgb;
    
    float diffuse = max(0.0, dot(lightDir, normal));
    float ambient = 0.4;
    
    vec3 halfwayDir = normalize(lightDir - axisZ);
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 100.0);
    
    vec3 result = color * (ambient + diffuse * 0.6) + vec3(1.0) * specular * 0.5;
    outColor = vec4(result, 1.0);
}
)";

// Ball shader - textured spheres with quaternion rotation
static const char* ballVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aRadius;
layout (location = 2) in vec3 aColor;
layout (location = 3) in vec4 aQuat;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform float pointScale;

out vec3 fragPos;
out float radius;
out vec3 eyePos;
out vec4 quat;
out vec3 viewRight;

void main() {
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    fragPos = aPos;
    radius = aRadius;
    eyePos = eyeSpacePos.xyz;
    quat = aQuat;
    viewRight = normalize(vec3(viewMat[0][0], viewMat[1][0], viewMat[2][0]));
    
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = aRadius * (pointScale / dist);
}
)";

static const char* ballFragmentShader = R"(
#version 330 core
const float PI = 3.14159265359;

uniform vec3 viewPos;
uniform vec3 lightDir;
uniform sampler2D pointTexture;
uniform bool useTexture;

in vec3 fragPos;
in float radius;
in vec3 eyePos;
in vec4 quat;
in vec3 viewRight;

out vec4 outColor;

vec3 qtransform(vec4 q, vec3 v) {
    return v + 2.0 * cross(cross(v, q.xyz) + q.w * v, q.xyz);
}

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    coord.y = -coord.y;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    float h = sqrt(1.0 - r2);
    
    vec3 axisZ = normalize(fragPos - viewPos);
    vec3 axisX = normalize(viewRight);
    vec3 axisY = cross(axisX, axisZ);
    
    vec3 localPos = radius * (coord.x * axisX + coord.y * axisY - h * axisZ);
    vec3 normal = normalize(localPos);
    
    vec3 rotNormal = qtransform(quat, vec3(normal.x, -normal.y, normal.z));
    
    vec3 color;
    if (useTexture) {
        float phi = atan(rotNormal.z, rotNormal.x);
        float u = phi / (2.0 * PI) + 0.5;
        float v = acos(clamp(rotNormal.y, -1.0, 1.0)) / PI;
        color = texture(pointTexture, vec2(u, v)).rgb;
    } else {
        // Beach ball pattern
        float angle = atan(rotNormal.z, rotNormal.x);
        int segment = int((angle + PI) / (PI * 2.0) * 6.0) % 6;
        if (abs(rotNormal.y) > 0.95) segment = 5;
        
        vec3 colors[6];
        colors[0] = vec3(1.0, 0.2, 0.2);
        colors[1] = vec3(1.0, 0.5, 0.2);
        colors[2] = vec3(1.0, 0.95, 0.2);
        colors[3] = vec3(0.2, 1.0, 0.2);
        colors[4] = vec3(0.2, 0.2, 1.0);
        colors[5] = vec3(1.0, 1.0, 1.0);
        color = colors[segment];
    }
    
    float diffuse = max(0.0, dot(lightDir, normal));
    float ambient = 0.4;
    
    vec3 halfwayDir = normalize(lightDir - axisZ);
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 100.0);
    
    vec3 result = color * (ambient + diffuse * 0.6) + vec3(1.0) * specular * 0.5;
    outColor = vec4(result, 1.0);
}
)";

// Shadow shader - depth only
static const char* shadowVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aRadius;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform float pointScale;

void main() {
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    gl_PointSize = aRadius * pointScale;
}
)";

static const char* shadowFragmentShader = R"(
#version 330 core
out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    if (dot(coord, coord) > 1.0) discard;
    float depth = gl_FragCoord.z;
    fragColor = vec4(depth, depth, depth, 1.0);
}
)";

// Particle shader - solid colored spheres with lifetime, uniform radius
static const char* particleVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 2) in vec3 aColor;
layout (location = 4) in float aLifetime;

uniform mat4 projectionMat;
uniform mat4 viewMat;
uniform float pointScale;
uniform float uniformRadius;

out vec3 fragColor;
out vec3 eyePos;
out float lifetime;

void main() {
    vec4 eyeSpacePos = viewMat * vec4(aPos, 1.0);
    gl_Position = projectionMat * eyeSpacePos;
    
    fragColor = aColor;
    eyePos = eyeSpacePos.xyz;
    lifetime = aLifetime;
    
    float dist = length(eyeSpacePos.xyz);
    gl_PointSize = uniformRadius * (pointScale / dist);
}
)";

static const char* particleFragmentShader = R"(
#version 330 core
uniform vec3 lightDir;

in vec3 fragColor;
in vec3 eyePos;
in float lifetime;

out vec4 outColor;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    coord.y = -coord.y;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) discard;
    
    float h = sqrt(1.0 - r2);
    vec3 normal = normalize(vec3(coord.x, coord.y, h));
    
    float diffuse = max(0.0, dot(lightDir, normal));
    float ambient = 0.4;
    
    vec3 halfwayDir = normalize(lightDir + vec3(0.0, 0.0, 1.0));
    float specular = pow(max(dot(normal, halfwayDir), 0.0), 32.0);
    
    vec3 result = fragColor * (ambient + diffuse * 0.6) + vec3(1.0) * specular * 0.3;
    outColor = vec4(result, 1.0);
}
)";

PointRenderer::PointRenderer() {}

PointRenderer::~PointRenderer() {
    cleanup();
}

bool PointRenderer::init(int maxPts, const AttribLayout& attribLayout) {
    maxPoints = maxPts;
    layout = attribLayout;
    
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, maxPoints * layout.strideFloats * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    setupVertexAttributes();
    glBindVertexArray(0);
    
    initShaders();
    return true;
}

void PointRenderer::setupVertexAttributes() {
    int stride = layout.strideFloats * sizeof(float);
    
    // Position (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)(layout.posOffset * sizeof(float)));
    glEnableVertexAttribArray(0);
    
    // Radius (location 1) - optional for particle mode
    if (layout.radiusOffset >= 0) {
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, (void*)(layout.radiusOffset * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    
    // Color (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(layout.colorOffset * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    // Quaternion (location 3) - optional
    if (layout.quatOffset >= 0) {
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, (void*)(layout.quatOffset * sizeof(float)));
        glEnableVertexAttribArray(3);
    }
    
    // Lifetime (location 4) - optional for particle mode
    if (layout.lifetimeOffset >= 0) {
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, (void*)(layout.lifetimeOffset * sizeof(float)));
        glEnableVertexAttribArray(4);
    }
}

void PointRenderer::initShaders() {
    shaderBasic = RenderUtils::createShaderProgram(basicVertexShader, basicFragmentShader);
    shaderTextured = RenderUtils::createShaderProgram(texturedVertexShader, texturedFragmentShader);
    shaderBall = RenderUtils::createShaderProgram(ballVertexShader, ballFragmentShader);
    shaderParticle = RenderUtils::createShaderProgram(particleVertexShader, particleFragmentShader);
    shaderShadow = RenderUtils::createShaderProgram(shadowVertexShader, shadowFragmentShader);
}

void PointRenderer::resize(int newMaxPoints) {
    maxPoints = newMaxPoints;
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, maxPoints * layout.strideFloats * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PointRenderer::initShadowBuffer(int width, int height) {
    if (shadowFBO != 0 && shadowWidth == width && shadowHeight == height) {
        return;
    }
    
    if (shadowFBO != 0) {
        glDeleteFramebuffers(1, &shadowFBO);
        glDeleteTextures(1, &shadowTexture);
        glDeleteRenderbuffers(1, &shadowRBO);
    }
    
    shadowWidth = width;
    shadowHeight = height;
    
    glGenFramebuffers(1, &shadowFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    
    glGenTextures(1, &shadowTexture);
    glBindTexture(GL_TEXTURE_2D, shadowTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, shadowTexture, 0);
    
    glGenRenderbuffers(1, &shadowRBO);
    glBindRenderbuffer(GL_RENDERBUFFER, shadowRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, shadowRBO);
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void PointRenderer::getLightMatrix(float* outMatrix, const Vec3& lightDir, const Bounds3& sceneBounds) const {
    Vec3 lightDirection = lightDir.normalized();
    float roomDiag = (sceneBounds.maximum - sceneBounds.minimum).magnitude();
    Vec3 lightPos = -lightDirection * (roomDiag * 2.0f);
    Vec3 target(0.0f, 0.0f, 0.0f);
    
    Vec3 forward = (target - lightPos).normalized();
    Vec3 right = Vec3(0, 1, 0).cross(forward).normalized();
    Vec3 up = forward.cross(right);
    
    float view[16];
    view[0] = right.x;    view[4] = right.y;    view[8] = right.z;     view[12] = -right.dot(lightPos);
    view[1] = up.x;       view[5] = up.y;       view[9] = up.z;        view[13] = -up.dot(lightPos);
    view[2] = -forward.x; view[6] = -forward.y; view[10] = -forward.z; view[14] = forward.dot(lightPos);
    view[3] = 0;          view[7] = 0;          view[11] = 0;          view[15] = 1;
    
    float halfSize = roomDiag * 0.6f;
    float nearClip = 0.1f;
    float farClip = roomDiag * 4.0f;
    float projection[16];
    RenderUtils::buildOrthographicMatrix(projection, halfSize, nearClip, farClip);
    
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            outMatrix[col * 4 + row] = 0.0f;
            for (int k = 0; k < 4; k++) {
                outMatrix[col * 4 + row] += projection[k * 4 + row] * view[col * 4 + k];
            }
        }
    }
}

void PointRenderer::render(Camera* camera, int count, Mode mode, GLuint texture,
                           const Vec3& lightDir, int viewportWidth, int viewportHeight,
                           float uniformRadius) {
    if (!camera || count <= 0) return;
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    GLuint shader = shaderBasic;
    if (mode == Mode::Textured) shader = shaderTextured;
    else if (mode == Mode::Ball) shader = shaderBall;
    else if (mode == Mode::Particle) shader = shaderParticle;
    
    glUseProgram(shader);
    
    // Build matrices from camera
    float view[16], projection[16];
    Vec3 camPos = camera->pos;
    Vec3 camTarget = camera->pos + camera->forward;
    
    RenderUtils::buildViewMatrix(view, camPos.x, camPos.y, camPos.z,
                                 camTarget.x, camTarget.y, camTarget.z,
                                 camera->up.x, camera->up.y, camera->up.z);
    
    float fov = camera->fov * 3.14159f / 180.0f;
    float aspect = (float)viewportWidth / viewportHeight;
    RenderUtils::buildProjectionMatrix(projection, fov, aspect, camera->nearClip, camera->farClip);
    
    glUniformMatrix4fv(glGetUniformLocation(shader, "projectionMat"), 1, GL_FALSE, projection);
    glUniformMatrix4fv(glGetUniformLocation(shader, "viewMat"), 1, GL_FALSE, view);
    glUniform1f(glGetUniformLocation(shader, "pointScale"), viewportHeight * projection[5]);
    
    // Transform light direction to view space
    Vec3 normalizedLight = lightDir.normalized();
    Vec3 viewSpaceLight(
        view[0] * normalizedLight.x + view[1] * normalizedLight.y + view[2] * normalizedLight.z,
        view[4] * normalizedLight.x + view[5] * normalizedLight.y + view[6] * normalizedLight.z,
        view[8] * normalizedLight.x + view[9] * normalizedLight.y + view[10] * normalizedLight.z
    );
    glUniform3f(glGetUniformLocation(shader, "lightDir"), viewSpaceLight.x, viewSpaceLight.y, viewSpaceLight.z);
    
    if (mode == Mode::Particle) {
        glUniform1f(glGetUniformLocation(shader, "uniformRadius"), uniformRadius);
    }
    else if (mode == Mode::Textured || mode == Mode::Ball) {
        glUniform3f(glGetUniformLocation(shader, "viewPos"), camPos.x, camPos.y, camPos.z);
        
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shader, "pointTexture"), 0);
        
        if (mode == Mode::Ball) {
            glUniform1i(glGetUniformLocation(shader, "useTexture"), texture != 0 ? 1 : 0);
        }
    }
    
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    
    glDisable(GL_POINT_SPRITE);
    glUseProgram(0);
}

void PointRenderer::renderShadowPass(int count, const Vec3& lightDir, const Bounds3& sceneBounds,
                                     int viewportWidth, int viewportHeight) {
    if (count <= 0) return;
    
    initShadowBuffer(viewportWidth, viewportHeight);
    
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    glViewport(0, 0, viewportWidth, viewportHeight);
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    glUseProgram(shaderShadow);
    
    float lightViewProj[16];
    getLightMatrix(lightViewProj, lightDir, sceneBounds);
    
    float roomDiag = (sceneBounds.maximum - sceneBounds.minimum).magnitude();
    float halfSize = roomDiag * 0.6f;
    float pointScale = (1.0f / halfSize) * viewportHeight;
    
    float identity[16];
    RenderUtils::setIdentity(identity);
    
    glUniformMatrix4fv(glGetUniformLocation(shaderShadow, "projectionMat"), 1, GL_FALSE, lightViewProj);
    glUniformMatrix4fv(glGetUniformLocation(shaderShadow, "viewMat"), 1, GL_FALSE, identity);
    glUniform1f(glGetUniformLocation(shaderShadow, "pointScale"), pointScale);
    
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    
    glDisable(GL_POINT_SPRITE);
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
}

void PointRenderer::cleanup() {
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (shaderBasic) { glDeleteProgram(shaderBasic); shaderBasic = 0; }
    if (shaderTextured) { glDeleteProgram(shaderTextured); shaderTextured = 0; }
    if (shaderBall) { glDeleteProgram(shaderBall); shaderBall = 0; }
    if (shaderParticle) { glDeleteProgram(shaderParticle); shaderParticle = 0; }
    if (shaderShadow) { glDeleteProgram(shaderShadow); shaderShadow = 0; }
    if (shadowFBO) { glDeleteFramebuffers(1, &shadowFBO); shadowFBO = 0; }
    if (shadowTexture) { glDeleteTextures(1, &shadowTexture); shadowTexture = 0; }
    if (shadowRBO) { glDeleteRenderbuffers(1, &shadowRBO); shadowRBO = 0; }
}
