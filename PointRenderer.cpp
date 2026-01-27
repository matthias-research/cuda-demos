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
    
    // Radius (location 1)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, (void*)(layout.radiusOffset * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Color (location 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(layout.colorOffset * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    // Quaternion (location 3) - optional
    if (layout.quatOffset >= 0) {
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, stride, (void*)(layout.quatOffset * sizeof(float)));
        glEnableVertexAttribArray(3);
    }
}

void PointRenderer::initShaders() {
    shaderBasic = RenderUtils::createShaderProgram(basicVertexShader, basicFragmentShader);
    shaderTextured = RenderUtils::createShaderProgram(texturedVertexShader, texturedFragmentShader);
    shaderBall = RenderUtils::createShaderProgram(ballVertexShader, ballFragmentShader);
    shaderShadow = RenderUtils::createShaderProgram(shadowVertexShader, shadowFragmentShader);
}

void PointRenderer::resize(int newMaxPoints) {
    maxPoints = newMaxPoints;
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, maxPoints * layout.strideFloats * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PointRenderer::render(Camera* camera, int count, Mode mode, GLuint texture,
                           const Vec3& lightDir, int viewportWidth, int viewportHeight) {
    if (!camera || count <= 0) return;
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    GLuint shader = shaderBasic;
    if (mode == Mode::Textured) shader = shaderTextured;
    else if (mode == Mode::Ball) shader = shaderBall;
    
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
    
    if (mode == Mode::Textured || mode == Mode::Ball) {
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

void PointRenderer::renderShadowPass(const float* lightViewProj, int count, float pointScale) {
    if (count <= 0) return;
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    
    glUseProgram(shaderShadow);
    
    // For orthographic shadow maps, view and projection are combined in lightViewProj
    float identity[16];
    RenderUtils::setIdentity(identity);
    
    glUniformMatrix4fv(glGetUniformLocation(shaderShadow, "projectionMat"), 1, GL_FALSE, lightViewProj);
    glUniformMatrix4fv(glGetUniformLocation(shaderShadow, "viewMat"), 1, GL_FALSE, identity);
    glUniform1f(glGetUniformLocation(shaderShadow, "pointScale"), pointScale);
    
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, count);
    glBindVertexArray(0);
    
    glDisable(GL_POINT_SPRITE);
    glUseProgram(0);
}

void PointRenderer::cleanup() {
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (shaderBasic) { glDeleteProgram(shaderBasic); shaderBasic = 0; }
    if (shaderTextured) { glDeleteProgram(shaderTextured); shaderTextured = 0; }
    if (shaderBall) { glDeleteProgram(shaderBall); shaderBall = 0; }
    if (shaderShadow) { glDeleteProgram(shaderShadow); shaderShadow = 0; }
}
