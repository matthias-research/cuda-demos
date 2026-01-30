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
        numCubes = 0;
        restDensity = 1.0f;
    }

    // don't use destructor, cuda would call it in the kernel!

    void free()
    {
		sortedParticleIds.free();
		particleCoords.free();
		hashCellFirst.free();
		hashCellLast.free();
		cubeOfParticle.free();
        cornerDensities.free();
		cornerNormals.free();
		cubeCoords.free();
        cubeEdgeLinks.free();
        edgeVertexNr.free();

        meshVertices.free();
        meshNormals.free();
        triIndices.free();
        clear();
    }

    int numParticles;
    float gridSpacing;
    float invGridSpacing;
    float worldOrig;
    int numCubes;
    float restDensity;

    // hash
    DeviceBuffer<int> sortedParticleIds; 
    DeviceBuffer<uint64_t> particleCoords;
    DeviceBuffer<int> hashCellFirst;
    DeviceBuffer<int> hashCellLast;
    DeviceBuffer<int> cubeOfParticle;
    DeviceBuffer<float> cornerDensities;
    DeviceBuffer<Vec3> cornerNormals;
    DeviceBuffer<uint64_t> cubeCoords;
    DeviceBuffer<int> cubeEdgeLinks;
    DeviceBuffer<int> edgeVertexNr;

    // mesh
    DeviceBuffer<Vec3> meshVertices;
    DeviceBuffer<Vec3> meshNormals;
    DeviceBuffer<int> triIndices;
};

// Kernels -------------------------------

__device__ inline unsigned int hashFunction(int xi, int yi, int zi)
{
    unsigned int h = abs((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) % MARCHING_CUBES_HASH_SIZE;
    return h;
}


__device__ inline void getPosCoords(const Vec3& pos, float invGridSpacing, float worldOrig, int& xi, int& yi, int& zi)
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
    getPosCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);

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
        data.cubeOfParticle[posNr] = 1; // cube changed, partial sum will do the rest
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


__global__ void kernel_addParticleDensities(MachingCubesSurfaceDeviceData data, int numParticles, const float* positions, int stride) 
{
    int posNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (posNr >= data.numParticles) 
        return;

    int pNr = data.sortedParticleIds[posNr];

    const float* posPtr = positions + pNr * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);

    int xi, yi, zi;
    getPosCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);
    Vec3 cellOrig = Vec3(xi * data.gridSpacing, yi * data.gridSpacing, zi * data.gridSpacing);

    int cubeNr = data.cubeOfParticle[pNr];

    // used for propagation of densities

    data.cubeCoords[cubeNr] = packCoords(xi, yi, zi);

    // use kernel to add densities to the cube corners

    float h = 1.5f * data.gridSpacing;
    float h2 = h * h;
    float kernelScale = 315.0f / (64.0f * 3.14159265f * h2 * h2 * h2 * h2 * h) / data.restDensity;    

    for (int i = 0; i < 8; i++)
    {
        int* cornerCoords = marchingCubeCorners[i];
        Vec3 cornerPos = cellOrig + Vec3((float)cornerCoords[0], (float)cornerCoords[1], (float)cornerCoords[2]) * data.gridSpacing;

        Vec3 r = pos - cornerPos;
        float r2 = r.magnitudeSquared();

        if (r2 < h2) 
        {
            float w = (h2 - r2);
            w = kernelScale * w * w * w;

            atomicAdd(&data.cornerDensities[8 * cubeNr + i], w);
            r.normalize();
            r *= -w;
            atomicAdd(&data.cornerNormals[8 * cubeNr + i].x, r.x);
            atomicAdd(&data.cornerNormals[8 * cubeNr + i].y, r.y);
            atomicAdd(&data.cornerNormals[8 * cubeNr + i].z, r.z);
        }   
    }
}


__global__ void kernel_sumCornerDensitiesAndFindEdgeLinks(MachingCubesSurfaceDeviceData data) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes) 
        return;

    // each corner of this cells sums up the densities of the adjacent cells

    uint64_t coord = data.cubeCoords[cubeNr];
    int xi, yi, zi;
    unpackCoords(coord, xi, yi, zi);

    int adjCubes[27];

    for (int i = 0; i < 27; i++)
    {
        if (i == 13)
        {
            adjCubes[i] = -1; // skip the center cell
            continue;
        }

        int xOffset = (i % 3) - 1;
        int yOffset = ((i / 3) % 3) - 1;
        int zOffset = ((i / 9) % 3) - 1;

        int adjXi = xi + xOffset;
        int adjYi = yi + yOffset;
        int adjZi = zi + zOffset;

        // find adjacent cell

        int adjCubeNr = -1;
        int h = hashFunction(adjXi, adjYi, adjZi);
        int first = data.hashCellFirst[h];
        int last = data.hashCellLast[h];

        // search the cube in the hash cell

        for (int j = first; j < last; j++)
        {
            uint64_t adjCoord = data.particleCoords[j];
            int testXi, testYi, testZi;
            unpackCoords(adjCoord, testXi, testYi, testZi);

            if (testXi == adjXi && testYi == adjYi && testZi == adjZi)
            {
                int id = data.sortedParticleIds[j];
                adjCubeNr = data.cubeOfParticle[id];
                break;
            }
        }

        adjCubes[i] = adjCubeNr;
    }

    // sum densities

    for (int cornerNr = 0; cornerNr < 8; cornerNr++)
    {
        for (int i = 0; i < 8; i++)
        {
            int adjCubeNr = adjCubes[cornerAdjCellNr[cornerNr][i]];
            if (adjCubeNr < 0)
                continue;

            int adjCornerNr = cornerAdjCornerNr[cornerNr][i];

            float adjDensity = data.cornerDensities[8 * adjCubeNr + adjCornerNr];
            Vec3 adjCornerNormal = data.cornerNormals[8 * adjCubeNr + adjCornerNr];

            // No atomics because we only manipule the thread's own cube

            data.cornerDensities[8 * cubeNr + cornerNr] += adjDensity;
            data.cornerNormals[8 * cubeNr + cornerNr] += adjCornerNormal;
        }
    }

    // find edge links

    for (int edgeNr = 0; edgeNr < 12; edgeNr++)
    {
        bool smallest = true;

        for (int i = 0; i < 4; i++)
        {
            int adjCubeNr = adjCubes[edgeAdjCellNr[edgeNr][i]];
            if (adjCubeNr < 0)
                continue;

            if (adjCubeNr < cubeNr)
                smallest = false;
        }

        // only the smallest creates a vertex
        // all others link to the smallest

        if (!smallest) {
            continue;

        for (int i = 0; i < 4; i++)
        {
            int adjCubeNr = adjCubes[edgeAdjCellNr[edgeNr][i]];
            if (adjCubeNr < 0)
                continue;

            int adjEdgeNr = edgeAdjEdgeNr[edgeNr][i];

            data.cubeEdgeLinks[12 * adjCubeNr + adjCubeNr] = 12 * cubeNr + edgeNr;
        }
        data.edgeVertexNr[12 * cubeNr + edgeNr] = 1; 
        data.cubeEdgeLinks[12 * cubeNr + edgeNr] = -1;
    }
}


__global__ void kernel_createCubeTriangles(MachingCubesSurfaceDeviceData data, bool countOnly) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes) 
        return;

    // each cube creates the triangles for its surface by marching through the iso-surface

    uint64_t coord = data.cubeCoords[cubeNr];
    int xi, yi, zi;
    unpackCoords(coord, xi, yi, zi);

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
    m_deviceData->cubeOfParticle.resize(numParticles, false);
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

    // Sort particles by coordinates

    thrust::device_ptr<uint64_t> particleCoords(m_deviceData->particleCoords.buffer);
    thrust::device_ptr<int> sortedParticleIds(m_deviceData->sortedParticleIds.buffer);
    thrust::sort_by_key(particleCoords, particleCoords + numParticles, sortedParticleIds);

    // Setup grid and hash 

    m_deviceData->cubeOfParticle.setZero();
    m_deviceData->hashCellFirst.setZero();
    m_deviceData->hashCellLast.setZero();

    kernel_setBoundaries<<<numParticles / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData, numParticles);
    cudaCheck(cudaGetLastError());

    thrust::device_ptr<int> cubeOfParticle(m_deviceData->cubeOfParticle.buffer);
    thrust::exclusive_scan(cubeOfParticle, cubeOfParticle + numParticles, cubeOfParticle);

    m_deviceData->cubeOfParticle.getDeviceObject(m_deviceData->numCubes, numParticles);

    // add the kernel densities to the cube corners

    m_deviceData->cubeCoords.resize(m_deviceData->numCubes, false);
    m_deviceData->cornerDensities.resize(8 * m_deviceData->numCubes, false);
    m_deviceData->cornerDensities.setZero();
    m_deviceData->cornerNormals.resize(8 * m_deviceData->numCubes, false);
    m_deviceData->cornerNormals.setZero();

    kernel_addParticleDensities<<<numParticles / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData, numParticles, particles, stride);
    cudaCheck(cudaGetLastError());

    // sum the corner densities of the adjacent cubes

    m_deviceData->cubeEdgeLinks.resize(12 * m_deviceData->numCubes, false);
    m_deviceData->edgeVertexNr.resize(12 * m_deviceData->numCubes + 1, false);
    m_deviceData->edgeVertexNr.setZero();

    kernel_sumCornerDensitiesAndFindEdgeLinks<<<m_deviceData->numCubes / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData);
    cudaCheck(cudaGetLastError());

    // create vertex slots

    thrust::device_ptr<int> edgeVertexNr(m_deviceData->edgeVertexNr.buffer);
    thrust::exclusive_scan(edgeVertexNr, edgeVertexNr + 12 * m_deviceData->numCubes + 1, edgeVertexNr);

    m_deviceData->edgeVertexNr.getDeviceObject(m_deviceData->numVertices, 12 * m_deviceData->numCubes);

    // create vertices




    // fill in the triangles

    kernel_createCubeTriangles<<<m_deviceData->numCubes / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData, false);


   // read back the mesh

    m_deviceData->meshVertices.get(m_vertices);
    m_deviceData->meshNormals.get(m_normals);
    m_deviceData->triIndices.get(m_triIndices);

    return true;
}
