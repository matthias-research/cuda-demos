#include "CudaHash.h"
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>



__device__ inline int hashPosition(const Vec3& pos, float gridSpacing, float worldOrig) 
{
    int xi = floorf((pos.x - worldOrig) / gridSpacing);
    int yi = floorf((pos.y - worldOrig) / gridSpacing);
    int zi = floorf((pos.z - worldOrig) / gridSpacing);
    
    return hashFunction(xi, yi, zi);
}

__global__ void kernel_fillHash(HashDeviceData data, const float* positions, int stride, const float* active) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numPoints) 
        return;
    
    // Check if particle is active (if active pointer is provided)
    if (active != nullptr) {
        const float* activePtr = active + idx * stride;
        if (*activePtr <= 0.0f) {
            // Mark as invalid by setting hash to a sentinel value (will be filtered out)
            data.hashVals[idx] = HASH_SIZE;  // Invalid hash value
            data.hashIds[idx] = idx;
            return;
        }
    }
    
    const float* posPtr = positions + idx * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);
    
    // Compute hash
    int h = hashPosition(pos, data.spacing, data.worldOrig);
    
    data.hashVals[idx] = h;
    data.hashIds[idx] = idx;
}

__global__ void kernel_setupHash(HashDeviceData data) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numPoints) 
        return;

    unsigned int h = data.hashVals[idx];
    
    // Find cell boundaries
    if (idx == 0) {
        data.hashCellFirst[h] = 0;
    } else {
        unsigned int prevH = data.hashVals[idx - 1];
        if (h != prevH) {
            data.hashCellFirst[h] = idx;
            data.hashCellLast[prevH] = idx;
        }
    }
    
    if (idx == data.numPoints - 1) {
        data.hashCellLast[h] = data.numPoints;
    }
}

__global__ void kernel_findNeighbors(HashDeviceData data, const float* positions, int stride, bool countOnly) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numPoints) return;
    
    const float* posPtr = positions + idx * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);
    
    int h = hashPosition(pos, data.spacing, data.worldOrig);

    float s = 1.0f / data.spacing;

    int xi = floorf((pos.x - data.worldOrig) * s);
    int yi = floorf((pos.y - data.worldOrig) * s);
    int zi = floorf((pos.z - data.worldOrig) * s);
    
    // Check neighboring cells (3x3x3 = 27 cells)

    int neighborNr = data.firstNeighbor[idx];

    for (int dx = -1; dx <= 1; dx++) 
    {
        for (int dy = -1; dy <= 1; dy++) 
        {
            for (int dz = -1; dz <= 1; dz++) 
            {
                int cellX = xi + dx;
                int cellY = yi + dy;
                int cellZ = zi + dz;
                
                unsigned int h = hashFunction(cellX, cellY, cellZ);
                
                int first = data.hashCellFirst[h];
                int last = data.hashCellLast[h];
                
                // Check all balls in this cell
                for (int i = first; i < last; i++) {
                    int otherIdx = data.hashIds[i];

                    if (countOnly)
                    {
                        data.firstNeighbor[idx] = data.firstNeighbor[idx] + 1;
                    }
                    else
                    {
                        data.neighbors[neighborNr] = otherIdx;
                        neighborNr++;
                    }
                }
            }
        }
    }
}


// Host functions 

void CudaHash::initialize() 
{
    if (!deviceData) 
        deviceData = std::make_shared<HashDeviceData>();
}


void CudaHash::cleanup() 
{
    if (deviceData) {
        deviceData->free();
        deviceData.reset();
    }
}

void CudaHash::fillHash(int numPoints, const float* positions, int stride, float spacing, const float* active)
{
    if (!deviceData)
        return;

    deviceData->numPoints = numPoints;
    deviceData->spacing = spacing;
    deviceData->worldOrig = -100.0f;

    deviceData->hashVals.resize(numPoints, false);
    deviceData->hashIds.resize(numPoints, false);
    deviceData->hashCellFirst.resize(HASH_SIZE, false);
    deviceData->hashCellLast.resize(HASH_SIZE, false);

    int numBlocks = (numPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kernel_fillHash<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData, positions, stride, active);
    cudaCheck(cudaGetLastError());

    // Sort by hash
    if (numPoints > 0) {
        thrust::device_ptr<int> hashVals(deviceData->hashVals.buffer);
        thrust::device_ptr<int> hashIds(deviceData->hashIds.buffer);
        thrust::sort_by_key(hashVals, hashVals + numPoints, hashIds);
    }

    deviceData->hashCellFirst.setZero();
    deviceData->hashCellLast.setZero();

    if (numPoints > 0) {
        int numBlocks = (numPoints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        kernel_setupHash<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData);
        cudaCheck(cudaGetLastError());
    }
}

void CudaHash::findNeighbors(int numPoints, const float* positions, int stride)
{
    if (!deviceData)
        return;

    deviceData->firstNeighbor.resize(numPoints + 1, false);
    deviceData->firstNeighbor.setZero();

    // count neighbors

    kernel_findNeighbors<<<numPoints / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*deviceData, positions, stride, true);
    cudaCheck(cudaGetLastError());

    thrust::device_ptr<int> firstNeighbor(deviceData->firstNeighbor.buffer);
    thrust::exclusive_scan(firstNeighbor, firstNeighbor + numPoints + 1, firstNeighbor);

    deviceData->firstNeighbor.getDeviceObject(deviceData->numNeighbors, numPoints);

    deviceData->neighbors.resize(deviceData->numNeighbors, false);

    // fill in neighbors

    kernel_findNeighbors<<<numPoints / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*deviceData, positions, stride, false);
    cudaCheck(cudaGetLastError());
}
