#pragma once

#include "Scene.h"
#include "Vec.h"
#include <memory>
#include "CudaUtils.h"
#include "BVH.h"

class BVHBuilder;


struct MeshesDeviceData
{
    void free()  // no destructor because cuda would call it in the kernels
    {
        numMeshes = 0;
        numMeshTriangles = 0;

        firstTriangle.free();
        vertices.free();
        triIds.free();
        meshBoundsLower.free();
        meshBoundsUpper.free();
        triBoundsLower.free();
        triBoundsUpper.free();
        trianglesBvh.free();
    }

    int numMeshes = 0;
    int numMeshTriangles = 0;

    DeviceBuffer<int> firstTriangle; 
    DeviceBuffer<Vec3> vertices;
    DeviceBuffer<int> triIds;

    DeviceBuffer<Vec4> triBoundsLower;
    DeviceBuffer<Vec4> triBoundsUpper;
    DeviceBuffer<Vec4> meshBoundsLower;
    DeviceBuffer<Vec4> meshBoundsUpper;
    BVH meshesBvh;
    BVH trianglesBvh;
};


class CudaMeshes
{
public:
    CudaMeshes() = default;
    ~CudaMeshes() = default;

    void initialize(const Scene* scene);
    void cleanup();

    bool rayCast(int numRays, float* positions, const Ray& ray, float* hits, int stride);

    const std::shared_ptr<MeshesDeviceData>& getDeviceData() const { return deviceData; }

private:
    std::shared_ptr<MeshesDeviceData> deviceData = nullptr;
    std::shared_ptr<BVHBuilder> bvhBuilder = nullptr;
};

