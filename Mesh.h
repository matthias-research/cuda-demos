#pragma once

#include <string>
#include <vector>
#include <GL/glew.h>
#include "Vec.h"

struct MeshData {
    std::vector<float> positions;     // x,y,z for each vertex
    std::vector<float> normals;       // nx,ny,nz for each vertex
    std::vector<float> texCoords;     // u,v for each vertex
    std::vector<unsigned int> indices; // triangle indices
    
    std::vector<unsigned char> textureData; // RGBA image data
    int textureWidth = 0;
    int textureHeight = 0;
    
    float transform[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; // Node transform from glTF
};

class Mesh {
public:
    Mesh();
    ~Mesh();
    
    // Load mesh from .glb file
    bool load(const std::string& filename);
    
    // Setup OpenGL buffers from already-populated MeshData (used by Scene)
    void setupGLFromData();
    
    // Get mesh data for CUDA access (positions, normals, indices)
    const MeshData& getData() const { return data; }
    
    // OpenGL resources
    GLuint getVAO() const { return vao; }
    GLuint getTexture() const { return texture; }
    unsigned int getIndexCount() const { return static_cast<unsigned int>(data.indices.size()); }
    
    // Cleanup
    void cleanup();
    
private:
    void setupGL();
    
    MeshData data;
    
    // OpenGL objects
    GLuint vao = 0;
    GLuint vbo = 0;      // positions, normals, texCoords interleaved
    GLuint ebo = 0;      // indices
    GLuint texture = 0;  // texture
};
