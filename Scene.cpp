#include "Scene.h"
#include <iostream>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

Scene::Scene() {
}

Scene::~Scene() {
    cleanup();
}

bool Scene::load(const std::string& filename) {
    // Clean up any existing meshes
    cleanup();
    
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
    
    if (model.meshes.empty()) {
        std::cerr << "No meshes found in glTF file" << std::endl;
        return false;
    }
    
    std::cout << "Loading scene with " << model.meshes.size() << " mesh(es)..." << std::endl;
    
    // Load all meshes and all their primitives
    for (size_t meshIdx = 0; meshIdx < model.meshes.size(); meshIdx++) {
        const tinygltf::Mesh& gltfMesh = model.meshes[meshIdx];
        
        for (size_t primIdx = 0; primIdx < gltfMesh.primitives.size(); primIdx++) {
            const tinygltf::Primitive& primitive = gltfMesh.primitives[primIdx];
            
            // Create a new Mesh object for this primitive
            Mesh* mesh = new Mesh();
            MeshData& data = const_cast<MeshData&>(mesh->getData());
            
            // Get node transform for this mesh
            float transform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
            for (const auto& node : model.nodes) {
                if (node.mesh == static_cast<int>(meshIdx)) {
                    if (!node.matrix.empty()) {
                        for (int i = 0; i < 16; i++) {
                            transform[i] = static_cast<float>(node.matrix[i]);
                        }
                    } else {
                        // Build from TRS
                        if (!node.scale.empty()) {
                            transform[0] *= node.scale[0];
                            transform[5] *= node.scale[1];
                            transform[10] *= node.scale[2];
                        }
                        
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
                            
                            float temp[16];
                            for (int i = 0; i < 16; i++) temp[i] = transform[i];
                            for (int row = 0; row < 3; row++) {
                                for (int col = 0; col < 3; col++) {
                                    transform[row * 4 + col] = 0;
                                    for (int k = 0; k < 3; k++) {
                                        transform[row * 4 + col] += temp[row * 4 + k] * m[k * 3 + col];
                                    }
                                }
                            }
                        }
                        
                        if (!node.translation.empty()) {
                            transform[12] = node.translation[0];
                            transform[13] = node.translation[1];
                            transform[14] = node.translation[2];
                        }
                    }
                    break;
                }
            }
            
            for (int i = 0; i < 16; i++) {
                data.transform[i] = transform[i];
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
                    float x = positions[i * 3 + 0];
                    float y = positions[i * 3 + 1];
                    float z = positions[i * 3 + 2];
                    
                    data.positions[i * 3 + 0] = transform[0] * x + transform[4] * y + transform[8] * z + transform[12];
                    data.positions[i * 3 + 1] = transform[1] * x + transform[5] * y + transform[9] * z + transform[13];
                    data.positions[i * 3 + 2] = transform[2] * x + transform[6] * y + transform[10] * z + transform[14];
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
                    float nx = normals[i * 3 + 0];
                    float ny = normals[i * 3 + 1];
                    float nz = normals[i * 3 + 2];
                    
                    data.normals[i * 3 + 0] = transform[0] * nx + transform[4] * ny + transform[8] * nz;
                    data.normals[i * 3 + 1] = transform[1] * nx + transform[5] * ny + transform[9] * nz;
                    data.normals[i * 3 + 2] = transform[2] * nx + transform[6] * ny + transform[10] * nz;
                    
                    float length = sqrtf(data.normals[i * 3 + 0] * data.normals[i * 3 + 0] +
                                        data.normals[i * 3 + 1] * data.normals[i * 3 + 1] +
                                        data.normals[i * 3 + 2] * data.normals[i * 3 + 2]);
                    if (length > 0.0f) {
                        data.normals[i * 3 + 0] /= length;
                        data.normals[i * 3 + 1] /= length;
                        data.normals[i * 3 + 2] /= length;
                    }
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
                    
                    if (image.component == 3) {
                        data.textureData.resize(image.width * image.height * 4);
                        for (int i = 0; i < image.width * image.height; i++) {
                            data.textureData[i * 4 + 0] = image.image[i * 3 + 0];
                            data.textureData[i * 4 + 1] = image.image[i * 3 + 1];
                            data.textureData[i * 4 + 2] = image.image[i * 3 + 2];
                            data.textureData[i * 4 + 3] = 255;
                        }
                    } else if (image.component == 4) {
                        data.textureData = image.image;
                    }
                }
            }
            
            // Setup OpenGL buffers for this mesh
            mesh->setupGLFromData();
            
            meshes.push_back(mesh);
            
            std::cout << "  Loaded mesh " << meshIdx << ", primitive " << primIdx 
                      << " (" << data.positions.size() / 3 << " vertices, " 
                      << data.indices.size() / 3 << " triangles)" << std::endl;
        }
    }
    
    std::cout << "Scene loaded: " << meshes.size() << " total primitive(s)" << std::endl;
    return true;
}

void Scene::cleanup() {
    for (Mesh* mesh : meshes) {
        mesh->cleanup();
        delete mesh;
    }
    meshes.clear();
}
