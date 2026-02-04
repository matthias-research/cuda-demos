#pragma once

#include "Vec.h"
#include <vector>
#include <memory>

struct MarchingCubesSurfaceDeviceData;
struct cudaGraphicsResource;

class MarchingCubesSurface
{
public:
    MarchingCubesSurface();
    ~MarchingCubesSurface();

    bool initialize(int numParticles, float gridSpacing, bool useBufferObjects);

    bool update(int numParticles, const float* particlePositions, int stride, bool onGpu);

    int getNumVertices() const;
    int getNumTriangles() const;

    void readBackMesh();

    const std::vector<Vec3>& getVertices() const
    {
        return m_vertices;
    }
    
    const std::vector<Vec3>& getNormals() const
    {
        return m_normals;
    }

    const std::vector<int>& getTriIndices() const
    {
        return m_triIndices;
    }

    unsigned int getVerticesVbo() const { return m_useBufferObjects ?  m_verticesVbo : 0; }
    unsigned int getNormalsVbo() const { return m_useBufferObjects ? m_normalsVbo : 0; }
    unsigned int getTriIndicesIbo() const { return m_useBufferObjects ? m_triIdsIbo : 0; }

    // Set the density threshold for the isosurface (should be ~restDensity * 0.5)
    void setDensityThreshold(float threshold);
    
    // Get debug info
    int getNumCubes() const;

    void free();

private:  
    bool m_initialized = false;

    std::shared_ptr<MarchingCubesSurfaceDeviceData> m_deviceData = nullptr;
    std::vector<Vec3> m_vertices;
    std::vector<Vec3> m_normals;
    std::vector<int> m_triIndices;

    bool m_useBufferObjects = false;
    unsigned int m_verticesVbo = 0;
    unsigned int m_normalsVbo = 0;
    unsigned int m_triIdsIbo = 0;

    cudaGraphicsResource* m_cudaVerticesVboResource = nullptr;
    cudaGraphicsResource* m_cudaNormalsVboResource = nullptr;
    cudaGraphicsResource* m_cudaTriIboResource = nullptr;

    // Capacity tracking to avoid reallocating VBOs every frame
    size_t m_verticesVboCapacity = 0;
    size_t m_normalsVboCapacity = 0;
    size_t m_triIdsIboCapacity = 0;
};
