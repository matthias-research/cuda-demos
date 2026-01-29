#pragma once

#include "Vec.h"
#include <vector>
#include <memory>

struct MachingCubesSurfaceDeviceData;

class MarchingCubesSurface
{
public:
    MarchingCubesSurface();
    ~MarchingCubesSurface();

    bool initialize(int numParticles, float gridSpacing);

    bool update(int numParticles, const float* particles, int stride);

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



private:  
    std::shared_ptr<MachingCubesSurfaceDeviceData> m_deviceData = nullptr;

    std::vector<Vec3> m_vertices;
    std::vector<Vec3> m_normals;
    std::vector<int> m_triIndices;
};
