#include "BoxesDemo.h"
#include <imgui.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// Vertex shader with Phong lighting
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

// Fragment shader with Phong lighting
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 objectColor;
uniform float ambientStrength;
uniform float specularStrength;
uniform float shininess;

void main()
{
    // Ambient
    vec3 ambient = ambientStrength * vec3(1.0);
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0);
    
    // Specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

// Cube vertices with normals (36 vertices for 6 faces)
float cubeVertices[] = {
    // positions          // normals
    // Back face
    -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
     1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
    // Front face
    -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
     1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
    // Left face
    -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f,
    // Right face
     1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f,
    // Bottom face
    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f,
     1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f,
     1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f,
    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f,
    -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f,
    // Top face
    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f,
     1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,
    -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,
    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f
};

// Helper function to create and compile shader
GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    // Check for errors
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
void setIdentity(float* mat) {
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
}

void setPerspective(float* mat, float fov, float aspect, float near, float far) {
    setIdentity(mat);
    float f = 1.0f / tan(fov * 0.5f);
    mat[0] = f / aspect;
    mat[5] = f;
    mat[10] = (far + near) / (near - far);
    mat[11] = -1.0f;
    mat[14] = (2.0f * far * near) / (near - far);
    mat[15] = 0.0f;
}

void setLookAt(float* mat, float eyeX, float eyeY, float eyeZ, 
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

void setRotationY(float* mat, float angle) {
    setIdentity(mat);
    float c = cos(angle);
    float s = sin(angle);
    mat[0] = c;  mat[2] = s;
    mat[8] = -s; mat[10] = c;
}

BoxesDemo::BoxesDemo() : vao(0), vbo(0), shaderProgram(0), fbo(0), renderTexture(0), 
                         fbWidth(0), fbHeight(0) {
    initCube();
    initShaders();
}

BoxesDemo::~BoxesDemo() {
    cleanupGL();
}

void BoxesDemo::initCube() {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void BoxesDemo::initShaders() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    // Check for linking errors
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader linking failed:\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void BoxesDemo::initFramebuffer(int width, int height) {
    if (fbo != 0 && fbWidth == width && fbHeight == height) {
        return; // Already initialized with correct size
    }
    
    // Clean up old framebuffer if exists
    if (fbo != 0) {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &renderTexture);
    }
    
    fbWidth = width;
    fbHeight = height;
    
    // Create framebuffer
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    
    // Create texture for rendering
    glGenTextures(1, &renderTexture);
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Attach texture to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTexture, 0);
    
    // Create renderbuffer for depth
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

void BoxesDemo::cleanupGL() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (shaderProgram) glDeleteProgram(shaderProgram);
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (renderTexture) glDeleteTextures(1, &renderTexture);
}

void BoxesDemo::update(float deltaTime) {
    rotation += rotationSpeed * deltaTime;
    if (rotation > 360.0f) rotation -= 360.0f;
}

bool BoxesDemo::raycast(const Vec3& orig, const Vec3& dir, float& t) {
    // TODO: Implement ray-box intersection
    // For now, return false (no hit)
    return false;
}

void BoxesDemo::render(uchar4* d_out, int width, int height) {
    if (!camera) return; // Safety check
    
    // Initialize framebuffer if needed
    initFramebuffer(width, height);
    
    // Render to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width, height);
    
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glUseProgram(shaderProgram);
    
    // Set up matrices
    float model[16], view[16], projection[16];
    setRotationY(model, rotation * 3.14159f / 180.0f);
    
    // Use Camera object for view
    Vec3 camPos = camera->pos;
    Vec3 camTarget = camera->pos + camera->forward;
    setLookAt(view, camPos.x, camPos.y, camPos.z, camTarget.x, camTarget.y, camTarget.z, camera->up.x, camera->up.y, camera->up.z);
    setPerspective(projection, camera->fov * 3.14159f / 180.0f, (float)width / height, 0.1f, 100.0f);
    
    // Set uniforms
    GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, model);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
    
    GLint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    GLint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    GLint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
    GLint ambientLoc = glGetUniformLocation(shaderProgram, "ambientStrength");
    GLint specularLoc = glGetUniformLocation(shaderProgram, "specularStrength");
    GLint shininessLoc = glGetUniformLocation(shaderProgram, "shininess");
    
    glUniform3f(lightPosLoc, lightPosX, lightPosY, lightPosZ);
    glUniform3f(viewPosLoc, camPos.x, camPos.y, camPos.z);
    glUniform3f(objectColorLoc, 0.9f, 0.5f, 0.2f); // Orange color
    glUniform1f(ambientLoc, ambientStrength);
    glUniform1f(specularLoc, specularStrength);
    glUniform1f(shininessLoc, shininess);
    
    // Draw cube
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
    
    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Read pixels from framebuffer and copy to CUDA buffer
    glBindTexture(GL_TEXTURE_2D, renderTexture);
    std::vector<unsigned char> pixels(width * height * 4);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    
    // Copy to CUDA buffer (flip Y axis)
    for (int y = 0; y < height; y++) {
        cudaMemcpy(d_out + y * width, pixels.data() + (height - 1 - y) * width * 4, width * 4, cudaMemcpyHostToDevice);
    }
    
    // Reset OpenGL state to ensure other demos work correctly
    glUseProgram(0);
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);
}

void BoxesDemo::renderUI() {
    ImGui::Text("Camera Controls:");
    ImGui::Text("  WASD: Move, Q/E: Up/Down");
    ImGui::Text("  Left Mouse: Orbit Point");
    ImGui::Text("  Middle Mouse: Pan");
    ImGui::Text("  Right Mouse: Rotate View");
    ImGui::Text("  Wheel: Zoom");
    
    ImGui::Separator();
    ImGui::Text("Animation:");
    ImGui::SliderFloat("Rotation Speed##rot", &rotationSpeed, 0.0f, 180.0f, "%.1f deg/s");
    
    ImGui::Separator();
    ImGui::Text("Lighting:");
    ImGui::SliderFloat("Light X##light", &lightPosX, -10.0f, 10.0f);
    ImGui::SliderFloat("Light Y##light", &lightPosY, -10.0f, 10.0f);
    ImGui::SliderFloat("Light Z##light", &lightPosZ, -10.0f, 10.0f);
    
    ImGui::Separator();
    ImGui::Text("Material:");
    ImGui::SliderFloat("Ambient##mat", &ambientStrength, 0.0f, 1.0f);
    ImGui::SliderFloat("Specular##mat", &specularStrength, 0.0f, 2.0f);
    ImGui::SliderFloat("Shininess##mat", &shininess, 2.0f, 256.0f);
    
    if (ImGui::Button("Reset View")) {
        if (camera) camera->resetView();
    }
}

void BoxesDemo::reset() {
    rotation = 0.0f;
    rotationSpeed = 45.0f;
    lightPosX = 5.0f;
    lightPosY = 5.0f;
    lightPosZ = 5.0f;
    ambientStrength = 0.1f;
    specularStrength = 0.5f;
    shininess = 32.0f;
    if (camera) camera->resetView();
}
