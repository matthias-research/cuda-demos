#include "Mesh.h"
#include <iostream>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

Mesh::Mesh() {
}

Mesh::~Mesh() {
    cleanup();
}

bool Mesh::load(const std::string& filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    
    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    
    if (!warn.empty()) {
        std::cout << "glTF Warning: " << warn << std::endl;
    }
    
    if (!err.empty()) {
        std::cerr << "glTF Error: " << err << std::endl;
    }
    
    if (!ret) {
        std::cerr << "Failed to load glTF: " << filename << std::endl;
        return false;
    }
    
    // For simplicity, just load the first mesh
    if (model.meshes.empty()) {
        std::cerr << "No meshes found in glTF file" << std::endl;
        return false;
    }
    
    const tinygltf::Mesh& mesh = model.meshes[0];
    if (mesh.primitives.empty()) {
        std::cerr << "No primitives found in mesh" << std::endl;
        return false;
    }
    
    const tinygltf::Primitive& primitive = mesh.primitives[0];
    
    // Get node transform (search for the node that references this mesh)
    for (const auto& node : model.nodes) {
        if (node.mesh == 0) { // First mesh
            if (!node.matrix.empty()) {
                // Matrix is provided directly (column-major)
                for (int i = 0; i < 16; i++) {
                    data.transform[i] = static_cast<float>(node.matrix[i]);
                }
            } else {
                // Build from translation, rotation, scale (TRS)
                // Identity matrix
                for (int i = 0; i < 16; i++) data.transform[i] = 0;
                data.transform[0] = data.transform[5] = data.transform[10] = data.transform[15] = 1;
                
                // Apply scale
                if (!node.scale.empty()) {
                    data.transform[0] *= node.scale[0];
                    data.transform[5] *= node.scale[1];
                    data.transform[10] *= node.scale[2];
                }
                
                // Apply rotation (quaternion to matrix)
                if (!node.rotation.empty()) {
                    float qx = node.rotation[0], qy = node.rotation[1];
                    float qz = node.rotation[2], qw = node.rotation[3];
                    
                    float m[9];
                    m[0] = 1 - 2*(qy*qy + qz*qz);
                    m[1] = 2*(qx*qy + qz*qw);
                    m[2] = 2*(qx*qz - qy*qw);
                    m[3] = 2*(qx*qy - qz*qw);
                    m[4] = 1 - 2*(qx*qx + qz*qz);
                    m[5] = 2*(qy*qz + qx*qw);
                    m[6] = 2*(qx*qz + qy*qw);
                    m[7] = 2*(qy*qz - qx*qw);
                    m[8] = 1 - 2*(qx*qx + qy*qy);
                    
                    // Multiply rotation into transform
                    float temp[16];
                    for (int i = 0; i < 16; i++) temp[i] = data.transform[i];
                    for (int row = 0; row < 3; row++) {
                        for (int col = 0; col < 3; col++) {
                            data.transform[row * 4 + col] = 0;
                            for (int k = 0; k < 3; k++) {
                                data.transform[row * 4 + col] += temp[row * 4 + k] * m[k * 3 + col];
                            }
                        }
                    }
                }
                
                // Apply translation
                if (!node.translation.empty()) {
                    data.transform[12] = node.translation[0];
                    data.transform[13] = node.translation[1];
                    data.transform[14] = node.translation[2];
                }
            }
            break;
        }
    }
    
    // Load positions
    if (primitive.attributes.find("POSITION") != primitive.attributes.end()) {
        int posAccessor = primitive.attributes.at("POSITION");
        const tinygltf::Accessor& accessor = model.accessors[posAccessor];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
        
        const float* positions = reinterpret_cast<const float*>(
            &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
        
        data.positions.resize(accessor.count * 3);
        for (size_t i = 0; i < accessor.count; i++) {
            // Apply transform to vertices for CUDA collision detection
            float x = positions[i * 3 + 0];
            float y = positions[i * 3 + 1];
            float z = positions[i * 3 + 2];
            
            data.positions[i * 3 + 0] = data.transform[0] * x + data.transform[4] * y + data.transform[8] * z + data.transform[12];
            data.positions[i * 3 + 1] = data.transform[1] * x + data.transform[5] * y + data.transform[9] * z + data.transform[13];
            data.positions[i * 3 + 2] = data.transform[2] * x + data.transform[6] * y + data.transform[10] * z + data.transform[14];
        }
    }
    
    // Load normals
    if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
        int normAccessor = primitive.attributes.at("NORMAL");
        const tinygltf::Accessor& accessor = model.accessors[normAccessor];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
        
        const float* normals = reinterpret_cast<const float*>(
            &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
        
        data.normals.resize(accessor.count * 3);
        for (size_t i = 0; i < accessor.count; i++) {
            // Apply transform rotation to normals (ignore translation)
            float nx = normals[i * 3 + 0];
            float ny = normals[i * 3 + 1];
            float nz = normals[i * 3 + 2];
            
            data.normals[i * 3 + 0] = data.transform[0] * nx + data.transform[4] * ny + data.transform[8] * nz;
            data.normals[i * 3 + 1] = data.transform[1] * nx + data.transform[5] * ny + data.transform[9] * nz;
            data.normals[i * 3 + 2] = data.transform[2] * nx + data.transform[6] * ny + data.transform[10] * nz;
        }
    }
    
    // Load texture coordinates
    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
        int uvAccessor = primitive.attributes.at("TEXCOORD_0");
        const tinygltf::Accessor& accessor = model.accessors[uvAccessor];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
        
        const float* texCoords = reinterpret_cast<const float*>(
            &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
        
        data.texCoords.resize(accessor.count * 2);
        for (size_t i = 0; i < accessor.count * 2; i++) {
            data.texCoords[i] = texCoords[i];
        }
    }
    
    // Load indices
    if (primitive.indices >= 0) {
        const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
        
        data.indices.resize(accessor.count);
        
        // Handle different index types
        if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            const uint16_t* indices = reinterpret_cast<const uint16_t*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            for (size_t i = 0; i < accessor.count; i++) {
                data.indices[i] = indices[i];
            }
        } else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            const uint32_t* indices = reinterpret_cast<const uint32_t*>(
                &buffer.data[bufferView.byteOffset + accessor.byteOffset]);
            for (size_t i = 0; i < accessor.count; i++) {
                data.indices[i] = indices[i];
            }
        }
    }
    
    // Load texture if material has one
    if (primitive.material >= 0) {
        const tinygltf::Material& material = model.materials[primitive.material];
        if (material.pbrMetallicRoughness.baseColorTexture.index >= 0) {
            int texIndex = material.pbrMetallicRoughness.baseColorTexture.index;
            const tinygltf::Texture& tex = model.textures[texIndex];
            const tinygltf::Image& image = model.images[tex.source];
            
            data.textureWidth = image.width;
            data.textureHeight = image.height;
            
            // Convert to RGBA if needed
            if (image.component == 3) {
                // RGB to RGBA
                data.textureData.resize(image.width * image.height * 4);
                for (int i = 0; i < image.width * image.height; i++) {
                    data.textureData[i * 4 + 0] = image.image[i * 3 + 0];
                    data.textureData[i * 4 + 1] = image.image[i * 3 + 1];
                    data.textureData[i * 4 + 2] = image.image[i * 3 + 2];
                    data.textureData[i * 4 + 3] = 255;
                }
            } else if (image.component == 4) {
                // Already RGBA
                data.textureData = image.image;
            }
        }
    }
    
    // Setup OpenGL buffers
    setupGL();
    
    return true;
}

void Mesh::setupGL() {
    // Create VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    
    // Create VBO with interleaved data (position, normal, texCoord)
    size_t vertexCount = data.positions.size() / 3;
    std::vector<float> interleavedData;
    interleavedData.reserve(vertexCount * 8); // 3 pos + 3 normal + 2 uv
    
    for (size_t i = 0; i < vertexCount; i++) {
        // Position
        interleavedData.push_back(data.positions[i * 3 + 0]);
        interleavedData.push_back(data.positions[i * 3 + 1]);
        interleavedData.push_back(data.positions[i * 3 + 2]);
        
        // Normal
        if (i * 3 < data.normals.size()) {
            interleavedData.push_back(data.normals[i * 3 + 0]);
            interleavedData.push_back(data.normals[i * 3 + 1]);
            interleavedData.push_back(data.normals[i * 3 + 2]);
        } else {
            interleavedData.push_back(0.0f);
            interleavedData.push_back(1.0f);
            interleavedData.push_back(0.0f);
        }
        
        // TexCoord
        if (i * 2 < data.texCoords.size()) {
            interleavedData.push_back(data.texCoords[i * 2 + 0]);
            interleavedData.push_back(data.texCoords[i * 2 + 1]);
        } else {
            interleavedData.push_back(0.0f);
            interleavedData.push_back(0.0f);
        }
    }
    
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, interleavedData.size() * sizeof(float), 
                 interleavedData.data(), GL_STATIC_DRAW);
    
    // Position attribute (location 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute (location 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), 
                         (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // TexCoord attribute (location 2)
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), 
                         (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    
    // Create EBO
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.indices.size() * sizeof(unsigned int),
                 data.indices.data(), GL_STATIC_DRAW);
    
    // Create texture if available
    if (!data.textureData.empty()) {
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.textureWidth, data.textureHeight,
                    0, GL_RGBA, GL_UNSIGNED_BYTE, data.textureData.data());
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        glGenerateMipmap(GL_TEXTURE_2D);
        
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    
    glBindVertexArray(0);
}

void Mesh::cleanup() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (texture) glDeleteTextures(1, &texture);
    
    vao = vbo = ebo = texture = 0;
}
