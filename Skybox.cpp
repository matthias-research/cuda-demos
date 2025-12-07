#include "Skybox.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Vertex shader - renders cube at camera position with max depth
static const char* skyboxVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    TexCoords = aPos;
    
    // Remove translation from view matrix (only rotation)
    mat4 rotView = mat4(mat3(view));
    vec4 pos = projection * rotView * vec4(aPos, 1.0);
    
    // Force depth to maximum (1.0) so skybox is always behind everything
    gl_Position = pos.xyww;
}
)";

// Fragment shader - samples cubemap
static const char* skyboxFragmentShader = R"(
#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;

void main()
{
    FragColor = texture(skybox, TexCoords);
}
)";

Skybox::Skybox() 
    : cubemapTexture(0), vao(0), vbo(0), shaderProgram(0), initialized(false) {
}

Skybox::~Skybox() {
    cleanup();
}

void Skybox::cleanup() {
    if (cubemapTexture) glDeleteTextures(1, &cubemapTexture);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    if (shaderProgram) glDeleteProgram(shaderProgram);
    
    cubemapTexture = 0;
    vbo = 0;
    vao = 0;
    shaderProgram = 0;
    initialized = false;
}

bool Skybox::createShaders() {
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &skyboxVertexShader, nullptr);
    glCompileShader(vertexShader);
    
    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        printf("Skybox vertex shader compilation failed: %s\n", infoLog);
        return false;
    }
    
    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &skyboxFragmentShader, nullptr);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        printf("Skybox fragment shader compilation failed: %s\n", infoLog);
        glDeleteShader(vertexShader);
        return false;
    }
    
    // Link shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        printf("Skybox shader program linking failed: %s\n", infoLog);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return false;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return true;
}

bool Skybox::createCubeGeometry() {
    // Cube vertices for skybox (centered at origin)
    float skyboxVertices[] = {
        // positions          
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };
    
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), skyboxVertices, GL_STATIC_DRAW);
    
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    
    glBindVertexArray(0);
    
    return true;
}

// Helper function to rotate face data 90 degrees clockwise
static void rotateFace90CW(unsigned char* data, int size) {
    unsigned char* temp = new unsigned char[size * size * 3];
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int srcIdx = (y * size + x) * 3;
            int dstIdx = (x * size + (size - 1 - y)) * 3;
            temp[dstIdx + 0] = data[srcIdx + 0];
            temp[dstIdx + 1] = data[srcIdx + 1];
            temp[dstIdx + 2] = data[srcIdx + 2];
        }
    }
    memcpy(data, temp, size * size * 3);
    delete[] temp;
}

// Helper function to flip face horizontally
static void flipFaceHorizontal(unsigned char* data, int size) {
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size / 2; x++) {
            int leftIdx = (y * size + x) * 3;
            int rightIdx = (y * size + (size - 1 - x)) * 3;
            for (int c = 0; c < 3; c++) {
                unsigned char temp = data[leftIdx + c];
                data[leftIdx + c] = data[rightIdx + c];
                data[rightIdx + c] = temp;
            }
        }
    }
}

// Helper function to flip face vertically
static void flipFaceVertical(unsigned char* data, int size) {
    for (int y = 0; y < size / 2; y++) {
        for (int x = 0; x < size; x++) {
            int topIdx = (y * size + x) * 3;
            int bottomIdx = ((size - 1 - y) * size + x) * 3;
            for (int c = 0; c < 3; c++) {
                unsigned char temp = data[topIdx + c];
                data[topIdx + c] = data[bottomIdx + c];
                data[bottomIdx + c] = temp;
            }
        }
    }
}

bool Skybox::loadFromBMP(const char* filepath) {
    // Open BMP file
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        printf("Failed to open skybox BMP file: %s\n", filepath);
        return false;
    }
    
    // Read BMP header (14 bytes)
    unsigned char header[54];
    if (fread(header, 1, 54, file) != 54) {
        printf("Not a valid BMP file (header too small)\n");
        fclose(file);
        return false;
    }
    
    // Check BMP signature
    if (header[0] != 'B' || header[1] != 'M') {
        printf("Not a valid BMP file (invalid signature)\n");
        fclose(file);
        return false;
    }
    
    // Read image info from header
    unsigned int dataOffset = *(int*)&(header[0x0A]);
    unsigned int imageSize = *(int*)&(header[0x22]);
    unsigned int width = *(int*)&(header[0x12]);
    unsigned int height = *(int*)&(header[0x16]);
    unsigned short bitsPerPixel = *(short*)&(header[0x1C]);
    
    // Validate format - support 24-bit and 32-bit
    if (bitsPerPixel != 24 && bitsPerPixel != 32) {
        printf("BMP must be 24-bit or 32-bit format (got %d bits per pixel)\n", bitsPerPixel);
        fclose(file);
        return false;
    }
    
    int bytesPerPixel = bitsPerPixel / 8;
    
    // Calculate face size - cross layout is 4x3 grid
    int faceSize = width / 4;
    if (width != faceSize * 4 || height != faceSize * 3) {
        printf("BMP dimensions must be 4:3 ratio for cubemap cross (got %dx%d)\n", width, height);
        fclose(file);
        return false;
    }
    
    // BMP rows are padded to multiples of 4 bytes
    int rowSize = width * 3;  // Output is always RGB (3 bytes)
    int paddedRowSize = (width * bytesPerPixel + 3) & ~3;  // Input may be 3 or 4 bytes per pixel
    
    // Calculate actual image size with padding
    if (imageSize == 0) {
        imageSize = paddedRowSize * height;
    }
    
    // Allocate memory for full image
    unsigned char* data = new unsigned char[imageSize];
    
    // Read pixel data
    fseek(file, dataOffset, SEEK_SET);
    if (fread(data, 1, imageSize, file) != imageSize) {
        printf("Failed to read BMP pixel data\n");
        delete[] data;
        fclose(file);
        return false;
    }
    
    fclose(file);
    
    // BMP stores BGR(A), we need RGB - also flip vertically and remove padding
    unsigned char* flipped = new unsigned char[width * height * 3];
    for (int y = 0; y < (int)height; y++) {
        for (int x = 0; x < (int)width; x++) {
            int srcIdx = y * paddedRowSize + x * bytesPerPixel;
            int dstIdx = (height - 1 - y) * rowSize + x * 3;
            flipped[dstIdx + 0] = data[srcIdx + 2]; // R (from B)
            flipped[dstIdx + 1] = data[srcIdx + 1]; // G
            flipped[dstIdx + 2] = data[srcIdx + 0]; // B (from R)
            // Ignore alpha channel if present (bytesPerPixel == 4)
        }
    }
    delete[] data;
    data = flipped;
    
    // Extract 6 cube faces from cross layout:
    //       [top]
    // [left][front][right][back]
    //      [bottom]
    //
    // OpenGL cubemap orientation:
    // +X = right, -X = left, +Y = top, -Y = bottom, +Z = front, -Z = back
    
    struct FaceInfo {
        int x, y;
        GLenum target;
    };
    
    FaceInfo faces[6] = {
        { 2, 1, GL_TEXTURE_CUBE_MAP_POSITIVE_X }, // +X right (cross: right)
        { 0, 1, GL_TEXTURE_CUBE_MAP_NEGATIVE_X }, // -X left (cross: left)
        { 1, 0, GL_TEXTURE_CUBE_MAP_POSITIVE_Y }, // +Y top (cross: top)
        { 1, 2, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y }, // -Y bottom (cross: bottom)
        { 1, 1, GL_TEXTURE_CUBE_MAP_POSITIVE_Z }, // +Z front (cross: front)
        { 3, 1, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z }  // -Z back (cross: back)
    };
    
    // Create cubemap texture
    glGenTextures(1, &cubemapTexture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    
    // Tell OpenGL that our RGB data is tightly packed (1-byte alignment)
    // Default is 4, which causes skewing if row width is not a multiple of 4 bytes
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    
    // Allocate buffer for one face
    unsigned char* faceData = new unsigned char[faceSize * faceSize * 3];
    
    // Extract and upload each face
    for (int i = 0; i < 6; i++) {
        int startX = faces[i].x * faceSize;
        int startY = faces[i].y * faceSize;
        
        printf("Extracting face %d at grid position [%d,%d] for target 0x%X\n", 
               i, faces[i].x, faces[i].y, faces[i].target);
        
        // Extract face pixels
        for (int y = 0; y < faceSize; y++) {
            for (int x = 0; x < faceSize; x++) {
                int srcIdx = ((startY + y) * width + (startX + x)) * 3;
                int dstIdx = (y * faceSize + x) * 3;
                faceData[dstIdx + 0] = data[srcIdx + 0];
                faceData[dstIdx + 1] = data[srcIdx + 1];
                faceData[dstIdx + 2] = data[srcIdx + 2];
            }
        }
        
        // Apply rotations/flips to match OpenGL cubemap expectations
        // These transformations depend on how the cross was authored
        switch (faces[i].target) {
            case GL_TEXTURE_CUBE_MAP_POSITIVE_X: // right
                // Test: no rotation
                break;
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_X: // left
                // Test: no rotation
                break;
            case GL_TEXTURE_CUBE_MAP_POSITIVE_Y: // top
                // Test: no rotation
                break;
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y: // bottom
                // Test: no rotation
                break;
            case GL_TEXTURE_CUBE_MAP_POSITIVE_Z: // front
                // Test: no rotation
                break;
            case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z: // back
                // Test: no rotation
                break;
        }
        
        // Upload to GPU
        glTexImage2D(faces[i].target, 0, GL_RGB, faceSize, faceSize, 0, 
                     GL_RGB, GL_UNSIGNED_BYTE, faceData);
    }
    
    delete[] faceData;
    delete[] data;
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    
    printf("Loaded skybox from %s (face size: %dx%d)\n", filepath, faceSize, faceSize);
    
    // Create shaders and geometry
    if (!createShaders()) {
        cleanup();
        return false;
    }
    
    if (!createCubeGeometry()) {
        cleanup();
        return false;
    }
    
    initialized = true;
    return true;
}

void Skybox::render(Camera* camera) {
    if (!initialized || !camera) return;
    
    // Change depth function so skybox renders at max depth
    glDepthFunc(GL_LEQUAL);
    
    glUseProgram(shaderProgram);
    
    // Set uniforms
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, camera->viewMat);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, camera->projMat);
    
    // Bind cubemap texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
    glUniform1i(glGetUniformLocation(shaderProgram, "skybox"), 0);
    
    // Render cube
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
    
    // Restore default depth function
    glDepthFunc(GL_LESS);
}
