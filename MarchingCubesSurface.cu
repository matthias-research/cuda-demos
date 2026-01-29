#define NOMINMAX
#include "MarchingCubesSurface.h"
#include "CudaUtils.h"
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <vector>
#include "MarchingCubesTable.h"

static const int MARCHING_CUBES_HASH_SIZE = 37000111;  

struct MachingCubesSurfaceDeviceData
{
    MachingCubesSurfaceDeviceData()
    {
        clear();
    }

    void clear()
    {
        numParticles = 0;
        gridSpacing = 0.0f;
        invGridSpacing = 0.0f;
        worldOrig = -1000.0f;
        numCells = 0;
        restDensity = 1.0f;
    }

    // don't use destructor, cuda would call it in the kernel!

    void free()
    {
		sortedParticleIds.free();
		hashCellFirst.free();
		hashCellLast.free();
		particleCoords.free();
        densityCoords.free();
        densities.free();
        densityNormals.free();
		normals.free();
        triIndices.free();
        clear();
    }

    int numParticles;
    float gridSpacing;
    float invGridSpacing;
    float worldOrig;
    int numCells;
    float restDensity;

    // hash
    DeviceBuffer<int> sortedParticleIds; 
    DeviceBuffer<uint64_t> particleCoords;
    DeviceBuffer<int> hashCellFirst;
    DeviceBuffer<int> hashCellLast;
    DeviceBuffer<int> cellOfParticle;
    DeviceBuffer<float> densities;
    DeviceBuffer<Vec3> densityNormals;
    DeviceBuffer<uint32_t> densityCoords;

    // mesh
    DeviceBuffer<Vec3> vertices;
    DeviceBuffer<Vec3> normals;
    DeviceBuffer<int> triIndices;
};

// Kernels -------------------------------

__device__ inline unsigned int hashFunction(int xi, int yi, int zi)
{
    unsigned int h = abs((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) % MARCHING_CUBES_HASH_SIZE;
    return h;
}


__device__ inline void getCellCoords(const Vec3& pos, float invGridSpacing, float worldOrig, int& xi, int& yi, int& zi)
{
    // force to positive integers
    xi = Max(0, (int)floorf((pos.x - worldOrig) * invGridSpacing));
    yi = Max(0, (int)floorf((pos.y - worldOrig) * invGridSpacing));
    zi = Max(0, (int)floorf((pos.z - worldOrig) * invGridSpacing));
}


__device__ uint64_t packCoords(int xi, int yi, int zi)
{
    // Pack into 64-bit value: assumes coordinates fit in ~21 bits each
    return (static_cast<uint64_t>(xi) << 42) |
            (static_cast<uint64_t>(yi) << 21) |
            (static_cast<uint64_t>(zi));
}


__device__ void unpackCoords(uint64_t coord, int& xi, int& yi, int& zi)
{
    xi = (int)(coord >> 42);
    yi = (int)((coord >> 21) & 0x1FFFFF);
    zi = (int)(coord & 0x1FFFFF);
}


__global__ void kernel_fillHash(MachingCubesSurfaceDeviceData data, int numParticles, const float* positions, int stride) 
{
    int pNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (pNr >= numParticles) 
        return;
    
    const float* posPtr = positions + pNr * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);
    
    int xi, yi, zi;
    getCellCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);

    data.particleCoords[pNr] = packCoords(xi, yi, zi);
    data.sortedParticleIds[pNr] = pNr;
}


__global__ void kernel_setBoundaries(MachingCubesSurfaceDeviceData data, int numParticles) 
{
    int posNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (posNr >= data.numParticles) 
        return;

    // cell boundary

    if (posNr > 0 && data.particleCoords[posNr] != data.particleCoords[posNr - 1])
    {
        data.cellOfParticle[posNr] = 1; // cell changed, partial sum will do the rest
    }

    // hash boundary

    int xi, yi, zi;
    unpackCoords(data.particleCoords[posNr], xi, yi, zi);
    int h = hashFunction(xi, yi, zi);

    if (posNr == 0) 
    {
        data.hashCellFirst[h] = 0;
    } 
    else 
    {
        unpackCoords(data.particleCoords[posNr - 1], xi, yi, zi);
        int prevH = hashFunction(xi, yi, zi);

        if (h != prevH) 
        {
            data.hashCellFirst[h] = posNr;
            data.hashCellLast[prevH] = posNr;
        }
    }
    
    if (posNr == data.numParticles - 1) 
    {
        data.hashCellLast[h] = data.numParticles;
    }
}


__global__ void kernel_calculateDensities(MachingCubesSurfaceDeviceData data, int numParticles, const float* positions, int stride) 
{
    int posNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (posNr >= data.numParticles) 
        return;

    int pNr = data.sortedParticleIds[posNr];

    const float* posPtr = positions + pNr * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);

    int xi, yi, zi;
    getCellCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);
    Vec3 cellOrig = Vec3(xi * data.gridSpacing, yi * data.gridSpacing, zi * data.gridSpacing);

    int cellNr = data.cellOfParticle[pNr];
    data.densityCoords[cellNr] = packCoords(xi, yi, zi);

    float h = 1.5f * data.gridSpacing;
    float h2 = h * h;
    float kernelScale = 315.0f / (64.0f * 3.14159265f * h2 * h2 * h2 * h2 * h) / data.restDensity;    

    for (int i = 0; i < 8; i++)
    {
        int* cornerCoords = marchingCubeCorners[i];
        Vec3 cornerPos = cellOrig + Vec3((float)cornerCoords[0], (float)cornerCoords[1], (float)cornerCoords[2]) * data.gridSpacing;

        Vec3 r = pos - cornerPos;
        float r2 = r.magnitudeSquared();
        if (r2 < h2) {
            float w = (h2 - r2);
            w = kernelScale * w * w * w;

            atomicAdd(&data.densities[8 * cellNr + i], w);
            r.normalize();
            r *= -w;
            atomicAdd(&data.densityNormals[8 * cellNr + i].x, r.x);
            atomicAdd(&data.densityNormals[8 * cellNr + i].y, r.y);
            atomicAdd(&data.densityNormals[8 * cellNr + i].z, r.z);
        }   
    }
}


__global__ void kernel__propagateDensities(MachingCubesSurfaceDeviceData data) 
{
    int cellNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cellNr >= data.numCells) 
        return;

    uint32_t coord = data.densityCoords[cellNr];
    int xi, yi, zi;
    unpackCoords(coord, xi, yi, zi);

    for (int i = 0; i < 27; i++)
    {
        if (i == 13)
            continue; // skip the center cell

        int xOffset = i % 3;
        int yOffset = (i / 3) % 3;
        int zOffset = (i / 9) % 3;

        int adjXi = xi + xOffset - 1;
        int adjYi = yi + yOffset - 1;
        int adjZi = zi + zOffset - 1;

        // find adjacent cell

        int adjCellNr = -1;
        int h = hashFunction(adjXi, adjYi, adjZi);
        int first = data.hashCellFirst[h];
        int last = data.hashCellLast[h];

        for (int j = first; j < last; j++)
        {
            uint32_t adjCoord = data.particleCoords[j];
            int id = data.sortedParticleIds[j];
            int cellNr = data.cellOfParticle[id];

            int testXi, testYi, testZi;
            unpackCoords(adjCoord, testXi, testYi, testZi);
            if (testXi == adjXi && testYi == adjYi && testZi == adjZi)
            {
                adjCellNr = cellNr;
                break;
            }
        }

        if (adjCellNr < 0)
            continue;
        
        // add density to the adjacent cell

        for (int j = 0; j < 8; j++)
        {
            int xi = marchingCubeCorners[j][0] + xOffset;
            int yi = marchingCubeCorners[j][1] + yOffset;
            int zi = marchingCubeCorners[j][2] + zOffset;

            float density = gridDensities[xi + yi * 3 + zi * 9];

            atomicAdd(&data.densities[8 * adjCellNr + j], density);
            atomicAdd(&data.densityNormals[8 * adjCellNr + j].x, density * r.x);
            atomicAdd(&data.densityNormals[8 * adjCellNr + j].y, density * r.y);
            atomicAdd(&data.densityNormals[8 * adjCellNr + j].z, density * r.z);
        }
    }
}


// Host functions -------------------------------

MarchingCubesSurface::MarchingCubesSurface()
{
}

MarchingCubesSurface::~MarchingCubesSurface()
{
    if (m_deviceData)
    {
        m_deviceData->free();
    }
}

bool MarchingCubesSurface::initialize(int numParticles, float gridSpacing)
{
    if (!m_deviceData)
    {
        m_deviceData = std::make_shared<MachingCubesSurfaceDeviceData>();
    }

    m_deviceData->gridSpacing = gridSpacing;
    m_deviceData->invGridSpacing = 1.0f / gridSpacing;
    m_deviceData->worldOrig = -1000.0f;
    m_deviceData->numParticles = numParticles;

    m_deviceData->hashCellFirst.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->hashCellLast.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->particleCoords.resize(numParticles, false);
    m_deviceData->sortedParticleIds.resize(numParticles, false);
    m_deviceData->cellOfParticle.resize(numParticles, false);
}   

bool MarchingCubesSurface::update(int numParticles, const float* particles, int stride)
{
    if (!m_deviceData)
    {
        return false;
    }

    if (m_deviceData->numParticles != numParticles)
    {
        return false;
    }

    kernel_fillHash<<<numParticles / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData, numParticles, particles, stride);
    cudaCheck(cudaGetLastError());

    // Sort by particle coords

    thrust::device_ptr<uint64_t> particleCoords(m_deviceData->particleCoords.buffer);
    thrust::device_ptr<int> sortedParticleIds(m_deviceData->sortedParticleIds.buffer);
    thrust::sort_by_key(particleCoords, particleCoords + numParticles, sortedParticleIds);

    // Setup grid and hash 

    m_deviceData->cellOfParticle.setZero();
    m_deviceData->hashCellFirst.setZero();
    m_deviceData->hashCellLast.setZero();

    kernel_setBoundaries<<<numParticles / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData, numParticles);
    cudaCheck(cudaGetLastError());

    thrust::device_ptr<int> cellOfParticle(m_deviceData->cellOfParticle.buffer);
    thrust::exclusive_scan(cellOfParticle, cellOfParticle + numParticles, cellOfParticle);

    m_deviceData->densityCoords.resize(m_deviceData->numCells, false);
    m_deviceData->densities.resize(8 * m_deviceData->numCells, false);
    m_deviceData->densities.setZero();
    m_deviceData->densityNormals.resize(8 * m_deviceData->numCells, false);
    m_deviceData->densityNormals.setZero();

    kernel_calculateDensities<<<numParticles / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData, numParticles, particles, stride);
    cudaCheck(cudaGetLastError());



   

    m_deviceData->vertices.get(m_vertices);
    m_deviceData->normals.get(m_normals);
    m_deviceData->triIndices.get(m_triIndices);

    return true;
}

