#include "MarchingCubesRenderer.h"
#include "MarchingCubesSurface.h"
#include "RenderUtils.h"
#include "Camera.h"
#include <cstdio>
#include <cmath>

// Vertex shader for marching cubes mesh
static const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = aPos;
    Normal = aNormal;
    gl_Position = projection * view * vec4(aPos, 1.0);
}
)";

// Fragment shader with Phong lighting and Fresnel
static const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightDir;
uniform vec3 viewPos;
uniform vec3 baseColor;
uniform float ambientStrength;
uniform float specularStrength;
uniform float shininess;
uniform float fresnelPower;
uniform float alpha;

void main()
{
    // Normalize the normal
    vec3 norm = normalize(Normal);
    
    // Ensure normal faces camera (for transparent surfaces)
    vec3 viewDir = normalize(viewPos - FragPos);
    if (dot(norm, viewDir) < 0.0) {
        norm = -norm;
    }
    
    // Ambient
    vec3 ambient = ambientStrength * baseColor;
    
    // Diffuse (directional light)
    vec3 lightDirection = normalize(lightDir);
    float diff = max(dot(norm, lightDirection), 0.0);
    vec3 diffuse = diff * baseColor;
    
    // Specular (Blinn-Phong)
    vec3 halfVec = normalize(lightDirection + viewDir);
    float spec = pow(max(dot(norm, halfVec), 0.0), shininess);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    // Fresnel effect (more reflective at grazing angles)
    float NdotV = max(dot(norm, viewDir), 0.0);
    float fresnel = pow(1.0 - NdotV, fresnelPower);
    vec3 fresnelColor = fresnel * vec3(0.8, 0.9, 1.0) * 0.4;
    
    // Combine lighting
    vec3 result = ambient + diffuse + specular + fresnelColor;
    
    // Apply alpha with fresnel boost at edges
    float finalAlpha = alpha + fresnel * (1.0 - alpha) * 0.5;
    
    FragColor = vec4(result, finalAlpha);
}
)";

MarchingCubesRenderer::MarchingCubesRenderer() {}

MarchingCubesRenderer::~MarchingCubesRenderer() {
    cleanup();
}

bool MarchingCubesRenderer::init() {
    shaderProgram = RenderUtils::createShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (shaderProgram == 0) {
        fprintf(stderr, "MarchingCubesRenderer: Failed to create shader program\n");
        return false;
    }
    return true;
}

void MarchingCubesRenderer::setupVAO(GLuint verticesVbo, GLuint normalsVbo) {
    if (currentVerticesVbo == verticesVbo && currentNormalsVbo == normalsVbo && vao != 0) {
        return;
    }

    if (vao != 0) {
        glDeleteVertexArrays(1, &vao);
    }

    currentVerticesVbo = verticesVbo;
    currentNormalsVbo = normalsVbo;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Position attribute (location 0)
    glBindBuffer(GL_ARRAY_BUFFER, verticesVbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute (location 1)
    glBindBuffer(GL_ARRAY_BUFFER, normalsVbo);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void MarchingCubesRenderer::render(Camera* camera, MarchingCubesSurface* surface,
                                    const Vec3& lightDir, int width, int height) {
    if (!camera || !surface || !shaderProgram) return;

    GLuint verticesVbo = surface->getVerticesVbo();
    GLuint normalsVbo = surface->getNormalsVbo();
    GLuint triIndicesIbo = surface->getTriIndicesIbo();
    int numTriangles = surface->getNumTriangles();

    if (verticesVbo == 0 || normalsVbo == 0 || triIndicesIbo == 0 || numTriangles <= 0) {
        return;
    }

    setupVAO(verticesVbo, normalsVbo);

    glUseProgram(shaderProgram);

    // Set matrix uniforms
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, camera->viewMat);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, camera->projMat);

    // Set lighting uniforms
    Vec3 camPos = camera->pos;
    Vec3 normalizedLight = lightDir.normalized();

    glUniform3f(glGetUniformLocation(shaderProgram, "lightDir"), 
                normalizedLight.x, normalizedLight.y, normalizedLight.z);
    glUniform3f(glGetUniformLocation(shaderProgram, "viewPos"), 
                camPos.x, camPos.y, camPos.z);

    // Material properties
    glUniform3f(glGetUniformLocation(shaderProgram, "baseColor"),
                material.baseColor.x, material.baseColor.y, material.baseColor.z);
    glUniform1f(glGetUniformLocation(shaderProgram, "ambientStrength"), material.ambientStrength);
    glUniform1f(glGetUniformLocation(shaderProgram, "specularStrength"), material.specularStrength);
    glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), material.shininess);
    glUniform1f(glGetUniformLocation(shaderProgram, "fresnelPower"), material.fresnelPower);
    glUniform1f(glGetUniformLocation(shaderProgram, "alpha"), material.alpha);

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Render both sides for fluid
    glDisable(GL_CULL_FACE);

    // Draw mesh
    glBindVertexArray(vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triIndicesIbo);
    glDrawElements(GL_TRIANGLES, numTriangles * 3, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // Restore state
    glDisable(GL_BLEND);
    glEnable(GL_CULL_FACE);
    glUseProgram(0);
}

void MarchingCubesRenderer::cleanup() {
    if (vao) {
        glDeleteVertexArrays(1, &vao);
        vao = 0;
    }
    if (shaderProgram) {
        glDeleteProgram(shaderProgram);
        shaderProgram = 0;
    }
    currentVerticesVbo = 0;
    currentNormalsVbo = 0;
}
