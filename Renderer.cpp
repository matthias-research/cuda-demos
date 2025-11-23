#include "Renderer.h"
#include <iostream>
#include <cmath>

// Vertex shader with texture support
const char* texturedVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec4 FragPosLightSpace;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 lightSpaceMatrix;

void main()
{
    FragPos = aPos;  // Use untransformed position since vertices are already in world space
    Normal = aNormal;  // Normals already transformed in mesh loading
    TexCoord = aTexCoord;
    FragPosLightSpace = lightSpaceMatrix * vec4(aPos, 1.0);
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

// Fragment shader with texture and Phong lighting
const char* texturedFragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord;
in vec4 FragPosLightSpace;

uniform sampler2D textureSampler;
uniform sampler2D shadowMap;
uniform bool hasTexture;
uniform bool useShadowMap;

uniform vec3 lightDir;
uniform vec3 viewPos;
uniform float ambientStrength;
uniform float specularStrength;
uniform float shininess;
uniform float shadowBias;

float calculateShadow(vec4 fragPosLightSpace)
{
    // Perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    
    // Check if outside shadow map bounds
    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
        return 1.0;  // No shadow
    
    // Get depth value from shadow map (stored in color channel as grayscale)
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    
    // Get depth of current fragment
    float currentDepth = projCoords.z;
    
    // Check whether current fragment is in shadow (inverted comparison)
    float shadow = (currentDepth - shadowBias) > closestDepth ? 1.0 : 0.0;
    
    return shadow;
}

void main()
{
    // Base color from texture or white
    vec3 baseColor = hasTexture ? texture(textureSampler, TexCoord).rgb : vec3(1.0);
    
    // Normalize the normal
    vec3 norm = normalize(Normal);
    
    // Ambient
    vec3 ambient = ambientStrength * baseColor;
    
    // Diffuse (directional light)
    vec3 lightDirection = normalize(lightDir);
    float diff = max(dot(norm, lightDirection), 0.0);
    vec3 diffuse = diff * baseColor;
    
    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDirection, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    // Calculate shadow
    float shadow = useShadowMap ? calculateShadow(FragPosLightSpace) : 1.0;
    
    // Combine lighting with shadow (ambient is not affected by shadow)
    vec3 result = ambient + shadow * (diffuse + specular);
    FragColor = vec4(result, 1.0);
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

// Matrix helper functions
static void setIdentity(float* mat) {
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
}

static void setPerspective(float* mat, float fov, float aspect, float near, float far) {
    setIdentity(mat);
    float f = 1.0f / tan(fov * 0.5f);
    mat[0] = f / aspect;
    mat[5] = f;
    mat[10] = (far + near) / (near - far);
    mat[11] = -1.0f;
    mat[14] = (2.0f * far * near) / (near - far);
    mat[15] = 0.0f;
}

static void setLookAt(float* mat, float eyeX, float eyeY, float eyeZ, 
                      float centerX, float centerY, float centerZ,
                      float upX, float upY, float upZ) {
    float fX = centerX - eyeX;
    float fY = centerY - eyeY;
    float fZ = centerZ - eyeZ;
    float len = sqrt(fX*fX + fY*fY + fZ*fZ);
    fX /= len; fY /= len; fZ /= len;
    
    float sX = fY * upZ - fZ * upY;
    float sY = fZ * upX - fX * upZ;
    float sZ = fX * upY - fY * upX;
    len = sqrt(sX*sX + sY*sY + sZ*sZ);
    sX /= len; sY /= len; sZ /= len;
    
    float uX = sY * fZ - sZ * fY;
    float uY = sZ * fX - sX * fZ;
    float uZ = sX * fY - sY * fX;
    
    mat[0] = sX;  mat[4] = sY;  mat[8] = sZ;   mat[12] = -(sX*eyeX + sY*eyeY + sZ*eyeZ);
    mat[1] = uX;  mat[5] = uY;  mat[9] = uZ;   mat[13] = -(uX*eyeX + uY*eyeY + uZ*eyeZ);
    mat[2] = -fX; mat[6] = -fY; mat[10] = -fZ; mat[14] = (fX*eyeX + fY*eyeY + fZ*eyeZ);
    mat[3] = 0;   mat[7] = 0;   mat[11] = 0;   mat[15] = 1;
}

static void setRotationX(float* mat, float angle) {
    setIdentity(mat);
    float c = cos(angle);
    float s = sin(angle);
    mat[5] = c;  mat[6] = -s;
    mat[9] = s;  mat[10] = c;
}

Renderer::Renderer() {
    setIdentity(modelMatrix);
    setIdentity(viewMatrix);
    setIdentity(projectionMatrix);
}

Renderer::~Renderer() {
    cleanup();
}

bool Renderer::init() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, texturedVertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, texturedFragmentShaderSource);
    
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader linking failed:\n" << infoLog << std::endl;
        return false;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return true;
}

void Renderer::setupMatrices(Camera* camera, int width, int height) {
    // Model matrix (identity for now, could be passed in for transformations)
    setIdentity(modelMatrix);
    
    // View matrix from camera
    if (camera) {
        Vec3 camPos = camera->pos;
        Vec3 camTarget = camera->pos + camera->forward;
        setLookAt(viewMatrix, 
                 camPos.x, camPos.y, camPos.z,
                 camTarget.x, camTarget.y, camTarget.z,
                 camera->up.x, camera->up.y, camera->up.z);
        
        // Projection matrix
        setPerspective(projectionMatrix, 
                      camera->fov * 3.14159f / 180.0f,
                      (float)width / height,
                      0.1f, 1000.0f);  // Increased far plane for large meshes
    }
}

void Renderer::renderMesh(const Mesh& mesh, Camera* camera, int width, int height, float scale, const float* modelTransform, float rotationX, const ShadowMap* shadowMap) {
    if (!shaderProgram || !camera) return;
    
    setupMatrices(camera, width, height);
    
    // Start with base transform
    if (modelTransform) {
        for (int i = 0; i < 16; i++) modelMatrix[i] = modelTransform[i];
    } else {
        setIdentity(modelMatrix);
    }
    
    // Apply rotation if needed (for glTF Y-up to Z-up conversion)
    if (rotationX != 0.0f) {
        float rotMat[16];
        setRotationX(rotMat, rotationX);
        
        // Multiply modelMatrix by rotation
        float temp[16];
        for (int i = 0; i < 16; i++) temp[i] = modelMatrix[i];
        
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                modelMatrix[row * 4 + col] = 0;
                for (int k = 0; k < 4; k++) {
                    modelMatrix[row * 4 + col] += temp[row * 4 + k] * rotMat[k * 4 + col];
                }
            }
        }
    }
    
    // Apply uniform scale to the upper-left 3x3 only
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            modelMatrix[i * 4 + j] *= scale;
        }
    }
    
    glUseProgram(shaderProgram);
    
    // Set matrix uniforms
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLint lightSpaceLoc = glGetUniformLocation(shaderProgram, "lightSpaceMatrix");
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, modelMatrix);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, viewMatrix);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projectionMatrix);
    
    // Set light space matrix (identity if no shadow map)
    if (shadowMap) {
        glUniformMatrix4fv(lightSpaceLoc, 1, GL_FALSE, shadowMap->lightMatrix);
    } else {
        float identityMatrix[16];
        setIdentity(identityMatrix);
        glUniformMatrix4fv(lightSpaceLoc, 1, GL_FALSE, identityMatrix);
    }
    
    // Set lighting uniforms
    Vec3 camPos = camera->pos;
    GLint lightDirLoc = glGetUniformLocation(shaderProgram, "lightDir");
    GLint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    GLint ambientLoc = glGetUniformLocation(shaderProgram, "ambientStrength");
    GLint specularLoc = glGetUniformLocation(shaderProgram, "specularStrength");
    GLint shininessLoc = glGetUniformLocation(shaderProgram, "shininess");
    
    glUniform3f(lightDirLoc, light.x, light.y, light.z);
    glUniform3f(viewPosLoc, camPos.x, camPos.y, camPos.z);
    glUniform1f(ambientLoc, material.ambientStrength);
    glUniform1f(specularLoc, material.specularStrength);
    glUniform1f(shininessLoc, material.shininess);
    
    // Set shadow map
    GLint useShadowMapLoc = glGetUniformLocation(shaderProgram, "useShadowMap");
    GLint shadowBiasLoc = glGetUniformLocation(shaderProgram, "shadowBias");
    
    if (shadowMap && shadowMap->texture != 0) {
        glUniform1i(useShadowMapLoc, 1);
        glUniform1f(shadowBiasLoc, shadowMap->bias);
        
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, shadowMap->texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "shadowMap"), 1);
    } else {
        glUniform1i(useShadowMapLoc, 0);
    }
    
    // Set texture
    GLint hasTextureLoc = glGetUniformLocation(shaderProgram, "hasTexture");
    if (mesh.getTexture() != 0) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mesh.getTexture());
        glUniform1i(glGetUniformLocation(shaderProgram, "textureSampler"), 0);
        glUniform1i(hasTextureLoc, 1);
    } else {
        glUniform1i(hasTextureLoc, 0);
    }
    
    // Draw mesh
    glBindVertexArray(mesh.getVAO());
    glDrawElements(GL_TRIANGLES, mesh.getIndexCount(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
    // Cleanup
    if (mesh.getTexture() != 0) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    if (shadowMap && shadowMap->texture != 0) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glUseProgram(0);
}

void Renderer::cleanup() {
    if (shaderProgram) {
        glDeleteProgram(shaderProgram);
        shaderProgram = 0;
    }
}
