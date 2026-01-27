#pragma once

#include <GL/glew.h>
#include <cmath>
#include <cstdio>
#include <string>

namespace RenderUtils {

inline GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "Shader compilation failed:\n%s\n", infoLog);
    }
    return shader;
}

inline GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        fprintf(stderr, "Shader linking failed:\n%s\n", infoLog);
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
}

inline GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    return linkProgram(vs, fs);
}

// Load BMP texture, returns 0 on failure
inline GLuint loadTexture(const std::string& filename) {
    if (filename.empty()) return 0;
    
    FILE* file;
    fopen_s(&file, filename.c_str(), "rb");
    if (!file) return 0;
    
    unsigned char header[54];
    if (fread(header, 1, 54, file) != 54 || header[0] != 'B' || header[1] != 'M') {
        fclose(file);
        return 0;
    }
    
    unsigned int dataOffset = *(int*)&header[0x0A];
    unsigned int width = *(int*)&header[0x12];
    unsigned int height = *(int*)&header[0x16];
    unsigned short bitsPerPixel = *(short*)&header[0x1C];
    
    if ((bitsPerPixel != 24 && bitsPerPixel != 32) || width == 0 || height == 0) {
        fclose(file);
        return 0;
    }
    
    int bytesPerPixel = bitsPerPixel / 8;
    int paddedRowSize = (width * bytesPerPixel + 3) & ~3;
    int rowSize = width * 3;
    
    unsigned char* rawData = new unsigned char[paddedRowSize * height];
    unsigned char* imageData = new unsigned char[rowSize * height];
    
    fseek(file, dataOffset, SEEK_SET);
    if (fread(rawData, 1, paddedRowSize * height, file) != paddedRowSize * height) {
        delete[] rawData;
        delete[] imageData;
        fclose(file);
        return 0;
    }
    fclose(file);
    
    // Convert BGR to RGB and flip vertically
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            int srcIdx = y * paddedRowSize + x * bytesPerPixel;
            int dstIdx = (height - 1 - y) * rowSize + x * 3;
            imageData[dstIdx + 0] = rawData[srcIdx + 2];
            imageData[dstIdx + 1] = rawData[srcIdx + 1];
            imageData[dstIdx + 2] = rawData[srcIdx + 0];
        }
    }
    delete[] rawData;
    
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, imageData);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    delete[] imageData;
    return texture;
}

// Create procedural checkerboard texture
inline GLuint createCheckerTexture(int size = 512, int checkerSize = 32) {
    unsigned char* data = new unsigned char[size * size * 3];
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int idx = (y * size + x) * 3;
            bool check = ((x / checkerSize) + (y / checkerSize)) % 2 == 0;
            data[idx + 0] = check ? 255 : 64;
            data[idx + 1] = check ? 128 : 32;
            data[idx + 2] = check ? 200 : 96;
        }
    }
    
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    delete[] data;
    return texture;
}

// Matrix utilities
inline void setIdentity(float* mat) {
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
}

inline void buildViewMatrix(float* mat, float eyeX, float eyeY, float eyeZ,
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

inline void buildProjectionMatrix(float* mat, float fovRadians, float aspect, float near, float far) {
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    float f = 1.0f / tan(fovRadians * 0.5f);
    mat[0] = f / aspect;
    mat[5] = f;
    mat[10] = (far + near) / (near - far);
    mat[11] = -1.0f;
    mat[14] = (2.0f * far * near) / (near - far);
}

inline void buildOrthographicMatrix(float* mat, float halfSize, float near, float far) {
    for (int i = 0; i < 16; i++) mat[i] = 0.0f;
    mat[0] = 1.0f / halfSize;
    mat[5] = 1.0f / halfSize;
    mat[10] = -2.0f / (far - near);
    mat[14] = -(far + near) / (far - near);
    mat[15] = 1.0f;
}

} // namespace RenderUtils
