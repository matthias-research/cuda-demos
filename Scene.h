#pragma once

#include <string>
#include <vector>
#include "Mesh.h"

// Scene class: Represents a complete 3D scene loaded from a GLB/GLTF file
// Contains multiple meshes, each with their own geometry and texture
class Scene {
public:
    Scene();
    ~Scene();
    
    // Load all meshes from a .glb file
    bool load(const std::string& filename);
    
    // Get all meshes in the scene
    const std::vector<Mesh*>& getMeshes() const { return meshes; }
    
    // Get number of meshes
    size_t getMeshCount() const { return meshes.size(); }
    
    // Cleanup all resources
    void cleanup();
    
private:
    std::vector<Mesh*> meshes;
};
