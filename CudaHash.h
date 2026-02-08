#pragma once

#include "Vec.h"
#include <memory>
#include "CudaUtils.h"

struct HashDeviceData
{
    void free()  // no destructor because cuda would call it in the kernels
    {
        numPoints = 0;
        spacing = 0.0f;
        worldOrig = 0.0f;
        hashVals.free();
        hashIds.free();
        hashCellFirst.free();
        hashCellLast.free();
    }

    int numNeighbors = 0;
    int numPoints = 0;
    float spacing = 0.0f;
    float worldOrig = 0.0f;
    DeviceBuffer<int> hashVals;
    DeviceBuffer<int> hashIds; 
    DeviceBuffer<int> hashCellFirst;
    DeviceBuffer<int> hashCellLast;

    DeviceBuffer<int> firstNeighbor;
    DeviceBuffer<int> neighbors;
};


static const int HASH_SIZE = 37000111;  


__device__ inline unsigned int hashFunction(int xi, int yi, int zi)
{
    unsigned int h = abs((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) % HASH_SIZE;
    return h;
}

class CudaHash
{
public:
    CudaHash() = default;
    ~CudaHash() = default;

    void initialize();
    void cleanup();

    void fillHash(int numPoints, const float* positions, int stride, float spacing, const float* active = nullptr);
    void findNeighbors(int numPoints, const float* positions, int stride);

    const std::shared_ptr<HashDeviceData>& getDeviceData() const { return deviceData; }

private:
    std::shared_ptr<HashDeviceData> deviceData = nullptr;
};

