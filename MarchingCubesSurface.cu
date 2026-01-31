#define NOMINMAX
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "MarchingCubesSurface.h"
#include "CudaUtils.h"
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <vector>
#include "MarchingCubesTable.h"

static const int MARCHING_CUBES_HASH_SIZE = 37000111;

struct MarchingCubesSurfaceDeviceData
{
    MarchingCubesSurfaceDeviceData() { clear(); }

    void clear()
    {
        numParticles = 0;
        gridSpacing = 0.0f;
        invGridSpacing = 0.0f;
        worldOrig = -1000.0f;
        numCubes = 0;
        restDensity = 1.0f;
        densityThreshold = 0.0f;
        numTriIds = 0;
        numVertices = 0;
    }

    void free()
    {
        particlePos.free();
        particleCubes.free();
        cubes.free();
        hashFirstCube.free();
        hashLastCube.free();
        firstExpandedCube.free();
        neighborBits.free();
        cornerDensities.free();
        cornerNormals.free();
        cubeEdgeLinks.free();
        clear();
    }

    int numParticles;
    int numCubes;
    float gridSpacing;
    float invGridSpacing;
    float worldOrig;
    float restDensity;
    float densityThreshold;
    int numTriIds;
    int numVertices;

    DeviceBuffer<float> particlePos;
    DeviceBuffer<uint64_t> particleCubes;
    DeviceBuffer<uint64_t> cubes;
    DeviceBuffer<int> hashFirstCube;
    DeviceBuffer<int> hashLastCube;
    DeviceBuffer<int> firstExpandedCube;
    DeviceBuffer<int> neighborBits;

    DeviceBuffer<float> cornerDensities;
    DeviceBuffer<Vec3> cornerNormals;
    DeviceBuffer<int> cubeEdgeLinks;
    DeviceBuffer<int> cubeFirstTriId;
    DeviceBuffer<int> edgeVertexNr;

    DeviceBuffer<Vec3> meshVertices;
    DeviceBuffer<Vec3> meshNormals;
    DeviceBuffer<int> triIndices;
};

// Kernels -------------------------------

__device__ inline unsigned int hashFunction(int xi, int yi, int zi)
{
    // Using unsigned to avoid undefined behavior with abs() and INT_MIN
    unsigned int uxi = (unsigned int)xi;
    unsigned int uyi = (unsigned int)yi;
    unsigned int uzi = (unsigned int)zi;
    return ((uxi * 92837111u) ^ (uyi * 689287499u) ^ (uzi * 283923481u)) % MARCHING_CUBES_HASH_SIZE;
}

__device__ inline void getPosCoords(const Vec3& pos, float invGridSpacing, float worldOrig, int& xi, int& yi, int& zi)
{
    xi = Max(0, (int)floorf((pos.x - worldOrig) * invGridSpacing));
    yi = Max(0, (int)floorf((pos.y - worldOrig) * invGridSpacing));
    zi = Max(0, (int)floorf((pos.z - worldOrig) * invGridSpacing));
}

__device__ uint64_t packCoords(int xi, int yi, int zi)
{
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

__global__ void kernel_createParticleCubes(MarchingCubesSurfaceDeviceData data, int numParticles, const float* positions, int stride) 
{
    int pNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (pNr >= numParticles) 
        return;
    
    const float* posPtr = positions + pNr * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);
    
    int xi, yi, zi;
    getPosCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);

    data.particleCubes[pNr] = packCoords(xi, yi, zi);
}


__global__ void kernel_setHashBoundaries(MarchingCubesSurfaceDeviceData data, bool particleCubes) 
{
    int numCubes = particleCubes ? data.numParticles : data.numCubes;
  
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= numCubes) 
        return;

    uint64_t* cubes = particleCubes ? data.particleCubes.buffer : data.cubes.buffer;

    int xi, yi, zi;
    unpackCoords(cubes[cubeNr], xi, yi, zi);
    int h = hashFunction(xi, yi, zi);

    if (cubeNr == 0) {
        data.hashFirstCube[h] = 0;
    } else {
        int prevXi, prevYi, prevZi;
        unpackCoords(cubes[cubeNr - 1], prevXi, prevYi, prevZi);
        int prevH = hashFunction(prevXi, prevYi, prevZi);

        if (h != prevH) {
            data.hashFirstCube[h] = cubeNr;
            data.hashLastCube[prevH] = cubeNr;
        }
    }
    
    if (cubeNr == numCubes - 1) {
        data.hashLastCube[h] = numCubes;
    }
}


__global__ void kernel_findExpandedShell(MarchingCubesSurfaceDeviceData data) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numParticles) return;

    // skip duplicates
    if (cubeNr > 0 && data.particleCubes[cubeNr] == data.particleCubes[cubeNr - 1]) 
        return;

    int cx, cy, cz;
    unpackCoords(data.particleCubes[cubeNr], cx, cy, cz);

    // test surrounding cubes

    int numNeighbors = 0;
    unsigned int neighborBits = 0;
    int neighborNr = 0;

    for (int xi = cx - 1; xi <= cx + 1; xi++) 
    {
        for (int yi = cy - 1; yi <= cy + 1; yi++) 
        {
            for (int zi = cz - 1; zi <= cz + 1; zi++) 
            {
                int64_t adjCoord = packCoords(xi, yi, zi);

                int adjCubeNr = -1;
                int h = hashFunction(xi, yi, zi);
                int first = data.hashFirstCube[h];
                int last = data.hashLastCube[h];

                for (int j = first; j < last; j++) {
                    if (data.particleCubes[j] == adjCoord) {
                        adjCubeNr = j;
                        break;
                    }
                }
                if (adjCubeNr >= 0) 
                {
                    neighborBits |= 1 << neighborNr;
                    numNeighbors++;
                }
                neighborNr++;
            }
        }
    }
    data.firstExpandedCube[cubeNr] = numNeighbors;
    data.neighborBits[cubeNr] = neighborBits;
}


__global__ void kernel_createExpandedShell(MarchingCubesSurfaceDeviceData data) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numParticles) return;

    int cx, cy, cz;
    unpackCoords(data.particleCubes[cubeNr], cx, cy, cz);
    int neighborBits = data.neighborBits[cubeNr];

    int neighborNr = 0;
    int pos = data.firstExpandedCube[cubeNr];

    for (int xi = cx - 1; xi <= cx + 1; xi++) 
    {
        for (int yi = cy - 1; yi <= cy + 1; yi++) 
        {
            for (int zi = cz - 1; zi <= cz + 1; zi++) 
            {
                if (neighborBits & (1 << neighborNr))
                {
                    data.cubes[pos] = packCoords(xi, yi, zi);
                    pos++;
                }
                neighborNr++;
            }
        }
    }
}


__global__ void kernel_addParticleDensities(MarchingCubesSurfaceDeviceData data, int numParticles, const float* positions, int stride) 
{
    int pNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (pNr >= numParticles) return;

    const float* posPtr = positions + pNr * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);

    int xi, yi, zi;
    getPosCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);

    float h_rad = 1.5f * data.gridSpacing;
    float h2 = h_rad * h_rad;
    float kernelScale = 315.0f / (64.0f * 3.14159265f * h2 * h2 * h2 * h2 * h_rad) / data.restDensity;

    // Particles must check all neighbor cubes (expanded shell) to add density to corners
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int adjXi = xi + dx;
                int adjYi = yi + dy;
                int adjZi = zi + dz;

                int hash = hashFunction(adjXi, adjYi, adjZi);
                int first = data.hashFirstCube[hash];
                int last = data.hashLastCube[hash];

                for (int j = first; j < last; j++) {
                    if (data.cubes[j] == packCoords(adjXi, adjYi, adjZi)) {
                        int cubeNr = j;
                        Vec3 cellOrig = Vec3(adjXi * data.gridSpacing, adjYi * data.gridSpacing, adjZi * data.gridSpacing) + Vec3(data.worldOrig, data.worldOrig, data.worldOrig);

                        for (int i = 0; i < 8; i++) {
                            int* cornerCoords = marchingCubeCorners[i];
                            Vec3 cornerPos = cellOrig + Vec3((float)cornerCoords[0], (float)cornerCoords[1], (float)cornerCoords[2]) * data.gridSpacing;

                            Vec3 r = pos - cornerPos;
                            float r2 = r.magnitudeSquared();

                            if (r2 < h2) {
                                float w = (h2 - r2);
                                w = kernelScale * w * w * w;
                                atomicAdd(&data.cornerDensities[8 * cubeNr + i], w);
                                float rLen = sqrtf(r2);
                                if (rLen > 1e-8f) {
                                    r *= (-w / rLen);
                                    atomicAdd(&data.cornerNormals[8 * cubeNr + i].x, r.x);
                                    atomicAdd(&data.cornerNormals[8 * cubeNr + i].y, r.y);
                                    atomicAdd(&data.cornerNormals[8 * cubeNr + i].z, r.z);
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
}

// ... [Existing: kernel_sumCornerDensitiesAndFindEdgeLinks, kernel_createCubeTriangles, kernel_createVerticesAndNormals] ...
// Note: These kernels remain largely the same, but now operate on data.numCubes (the expanded shell).

__global__ void kernel_sumCornerDensitiesAndFindEdgeLinks(MarchingCubesSurfaceDeviceData data) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes) return;

    uint64_t coord = data.cubes[cubeNr];
    int xi, yi, zi;
    unpackCoords(coord, xi, yi, zi);

    int adjCubes[27];
    for (int i = 0; i < 27; i++) {
        if (i == 13) { adjCubes[i] = -1; continue; }

        int adjXi = xi + (i % 3) - 1;
        int adjYi = yi + ((i / 3) % 3) - 1;
        int adjZi = zi + ((i / 9) % 3) - 1;

        int adjCubeNr = -1;
        int h = hashFunction(adjXi, adjYi, adjZi);
        for (int j = data.hashFirstCube[h]; j < data.hashLastCube[h]; j++) {
            if (data.cubes[j] == packCoords(adjXi, adjYi, adjZi)) {
                adjCubeNr = j;
                break;
            }
        }
        adjCubes[i] = adjCubeNr;
    }

    int marchingCubesCode = 0;
    for (int cornerNr = 0; cornerNr < 8; cornerNr++) {
        float density = data.cornerDensities[8 * cubeNr + cornerNr];
        Vec3 normal = data.cornerNormals[8 * cubeNr + cornerNr];

        for (int i = 0; i < 8; i++) {
            int adjCubeNr = adjCubes[cornerAdjCellNr[cornerNr][i]];
            if (adjCubeNr < 0) continue;

            int adjCornerNr = cornerAdjCornerNr[cornerNr][i];
            density += data.cornerDensities[8 * adjCubeNr + adjCornerNr];
            normal += data.cornerNormals[8 * adjCubeNr + adjCornerNr];
        }

        data.cornerDensities[8 * cubeNr + cornerNr] = density;
        data.cornerNormals[8 * cubeNr + cornerNr] = normal;
        if (density > data.densityThreshold) marchingCubesCode |= 1 << cornerNr;
    }

    data.cubeFirstTriId[cubeNr] = firstMarchingCubesId[marchingCubesCode + 1] - firstMarchingCubesId[marchingCubesCode];

    for (int edgeNr = 0; edgeNr < 12; edgeNr++) {
        bool smallest = true;
        for (int i = 0; i < 4; i++) {
            int adjCubeNr = adjCubes[edgeAdjCellNr[edgeNr][i]];
            if (adjCubeNr >= 0 && adjCubeNr < cubeNr) smallest = false;
        }

        if (smallest) {
            for (int i = 0; i < 4; i++) {
                int adjCubeNr = adjCubes[edgeAdjCellNr[edgeNr][i]];
                if (adjCubeNr >= 0) data.cubeEdgeLinks[12 * adjCubeNr + edgeAdjEdgeNr[edgeNr][i]] = 12 * cubeNr + edgeNr;
            }
            data.cubeEdgeLinks[12 * cubeNr + edgeNr] = -1;

            float d0 = data.cornerDensities[8 * cubeNr + marchingCubeEdges[edgeNr][0]];
            float d1 = data.cornerDensities[8 * cubeNr + marchingCubeEdges[edgeNr][1]];
            if ((d0 <= data.densityThreshold && d1 > data.densityThreshold) || (d0 > data.densityThreshold && d1 <= data.densityThreshold)) {
                data.edgeVertexNr[12 * cubeNr + edgeNr] = 1;
            }
        }
    }
}

__global__ void kernel_createCubeTriangles(MarchingCubesSurfaceDeviceData data, int* triIndices) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes) return;

    int edgeVertexId[12];
    for (int edgeNr = 0; edgeNr < 12; edgeNr++) {
        int link = data.cubeEdgeLinks[12 * cubeNr + edgeNr];
        edgeVertexId[edgeNr] = data.edgeVertexNr[link < 0 ? 12 * cubeNr + edgeNr : link];
    }

    int marchingCubesCode = 0;
    for (int cornerNr = 0; cornerNr < 8; cornerNr++) {
        if (data.cornerDensities[8 * cubeNr + cornerNr] > data.densityThreshold) marchingCubesCode |= 1 << cornerNr;
    }

    if (marchingCubesCode == 0) return;

    int tableStart = firstMarchingCubesId[marchingCubesCode];
    int outPos = data.cubeFirstTriId[cubeNr];
    for (int i = 0; i < (firstMarchingCubesId[marchingCubesCode + 1] - tableStart); i++) {
        triIndices[outPos + i] = edgeVertexId[marchingCubesIds[tableStart + i]];
    }
}

__global__ void kernel_createVerticesAndNormals(MarchingCubesSurfaceDeviceData data, Vec3* vertices, Vec3* normals) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes) return;

    int xi, yi, zi;
    unpackCoords(data.cubes[cubeNr], xi, yi, zi);
    Vec3 cubeOrig = Vec3(xi * data.gridSpacing, yi * data.gridSpacing, zi * data.gridSpacing) + Vec3(data.worldOrig, data.worldOrig, data.worldOrig);

    for (int edgeNr = 0; edgeNr < 12; edgeNr++) {
        if (data.cubeEdgeLinks[12 * cubeNr + edgeNr] >= 0) continue;

        int id0 = marchingCubeEdges[edgeNr][0];
        int id1 = marchingCubeEdges[edgeNr][1];
        float d0 = data.cornerDensities[8 * cubeNr + id0];
        float d1 = data.cornerDensities[8 * cubeNr + id1];

        if ((d0 <= data.densityThreshold && d1 > data.densityThreshold) || (d0 > data.densityThreshold && d1 <= data.densityThreshold)) {
            float t = (d1 != d0) ? Clamp((data.densityThreshold - d0) / (d1 - d0), 0.0f, 1.0f) : 0.5f;
            Vec3 p0 = cubeOrig + Vec3((float)marchingCubeCorners[id0][0], (float)marchingCubeCorners[id0][1], (float)marchingCubeCorners[id0][2]) * data.gridSpacing;
            Vec3 p1 = cubeOrig + Vec3((float)marchingCubeCorners[id1][0], (float)marchingCubeCorners[id1][1], (float)marchingCubeCorners[id1][2]) * data.gridSpacing;
            
            int vId = data.edgeVertexNr[12 * cubeNr + edgeNr];
            vertices[vId] = p0 + t * (p1 - p0);
            Vec3 n = data.cornerNormals[8 * cubeNr + id0] + t * (data.cornerNormals[8 * cubeNr + id1] - data.cornerNormals[8 * cubeNr + id0]);
            n.normalize();
            normals[vId] = n;
        }
    }
}

// Host functions -------------------------------

MarchingCubesSurface::MarchingCubesSurface() {}
MarchingCubesSurface::~MarchingCubesSurface() { free(); }

bool MarchingCubesSurface::initialize(int numParticles, float gridSpacing, bool useBufferObjects)
{
    m_vertices.clear(); m_normals.clear(); m_triIndices.clear();
    m_initialized = false;

    if (!m_deviceData) m_deviceData = std::make_shared<MarchingCubesSurfaceDeviceData>();

    m_deviceData->gridSpacing = gridSpacing;
    m_deviceData->invGridSpacing = 1.0f / gridSpacing;
    m_deviceData->worldOrig = -1000.0f;
    m_deviceData->numParticles = numParticles;

    m_deviceData->particleCubes.resize(numParticles, false);
    m_deviceData->neighborBits.resize(numParticles, false);
    m_deviceData->hashFirstCube.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->hashLastCube.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->firstExpandedCube.resize(numParticles, false);

    if (useBufferObjects) {
        if (m_verticesVbo == 0) {
            glGenBuffers(1, &m_verticesVbo);
            cudaGraphicsGLRegisterBuffer(&m_cudaVerticesVboResource, m_verticesVbo, cudaGraphicsMapFlagsWriteDiscard);
        }
        if (m_normalsVbo == 0) {
            glGenBuffers(1, &m_normalsVbo);
            cudaGraphicsGLRegisterBuffer(&m_cudaNormalsVboResource, m_normalsVbo, cudaGraphicsMapFlagsWriteDiscard);
        }
        if (m_triIdsIbo == 0) {
            glGenBuffers(1, &m_triIdsIbo);
            cudaGraphicsGLRegisterBuffer(&m_cudaTriIboResource, m_triIdsIbo, cudaGraphicsMapFlagsWriteDiscard);
        }
    }

    m_useBufferObjects = useBufferObjects;
    m_initialized = true;
    return true;
}   

bool MarchingCubesSurface::update(int numParticles, const float* particlePositions, int stride, bool onGpu)
{
    if (!m_deviceData || !m_initialized || m_deviceData->numParticles != numParticles) return false;

    const float* particles = particlePositions;
    if (!onGpu) {
        m_deviceData->particlePos.set(particlePositions, numParticles * stride);
        particles = m_deviceData->particlePos.buffer;
    }

    // 1. Create one cube for each particle

    kernel_createParticleCubes<<<(numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*m_deviceData, numParticles, particles, stride);
    cudaCheck(cudaGetLastError());

    thrust::device_ptr<uint64_t> particleCubesPtr(m_deviceData->particleCubes.buffer);
    thrust::sort(particleCubesPtr, particleCubesPtr + numParticles);

    m_deviceData->hashFirstCube.setZero();
    m_deviceData->hashLastCube.setZero();
    kernel_setHashBoundaries<<<(numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*m_deviceData, true);
    cudaCheck(cudaGetLastError());

    // 2. Find expanded shell

    m_deviceData->firstExpandedCube.resize(m_deviceData->numParticles + 1, false);
    m_deviceData->firstExpandedCube.setZero();
    m_deviceData->neighborBits.resize(m_deviceData->numParticles, false);
    m_deviceData->neighborBits.setZero();
    m_deviceData->cubes.resize(numParticles, false);

    kernel_findExpandedShell<<<(m_deviceData->numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*m_deviceData);
    cudaCheck(cudaGetLastError());

    thrust::device_ptr<int> firstExpandedCubePtr(m_deviceData->firstExpandedCube.buffer);
    thrust::exclusive_scan(firstExpandedCubePtr, firstExpandedCubePtr + m_deviceData->numParticles + 1, firstExpandedCubePtr);

    m_deviceData->firstExpandedCube.getDeviceObject(m_deviceData->numCubes, m_deviceData->numParticles);

    // 3. Create expanded shell, then sort and reduce to unique cubes

    m_deviceData->cubes.resize(m_deviceData->numCubes, false);
    kernel_createExpandedShell<<<(m_deviceData->numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*m_deviceData);
    cudaCheck(cudaGetLastError());

    thrust::device_ptr<uint64_t> expandedPtr(m_deviceData->cubes.buffer);
    thrust::sort(expandedPtr, expandedPtr + m_deviceData->numCubes);
    auto expandedEnd = thrust::unique(expandedPtr, expandedPtr + m_deviceData->numCubes);
    m_deviceData->numCubes = expandedEnd - expandedPtr;

    m_deviceData->hashFirstCube.setZero();
    m_deviceData->hashLastCube.setZero();
    kernel_setHashBoundaries<<<(m_deviceData->numCubes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*m_deviceData, false);
    cudaCheck(cudaGetLastError());

    // 6. Accumulate Density

    m_deviceData->cornerDensities.resize(8 * m_deviceData->numCubes, false);
    m_deviceData->cornerDensities.setZero();
    m_deviceData->cornerNormals.resize(8 * m_deviceData->numCubes, false);
    m_deviceData->cornerNormals.setZero();
    kernel_addParticleDensities<<<(numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*m_deviceData, numParticles, particles, stride);
    cudaCheck(cudaGetLastError());

    // 7. Surface Generation

    m_deviceData->cubeEdgeLinks.resize(12 * m_deviceData->numCubes, false);
    m_deviceData->cubeEdgeLinks.setZero();
    m_deviceData->edgeVertexNr.resize(12 * m_deviceData->numCubes + 1, false);
    m_deviceData->edgeVertexNr.setZero();
    m_deviceData->cubeFirstTriId.resize(m_deviceData->numCubes + 1, false);
    m_deviceData->cubeFirstTriId.setZero();

    kernel_sumCornerDensitiesAndFindEdgeLinks<<<(m_deviceData->numCubes + 255) / 256, 256>>>(*m_deviceData);
    cudaCheck(cudaGetLastError());

    // Scan for triangle and vertex counts
    thrust::device_ptr<int> triScanPtr(m_deviceData->cubeFirstTriId.buffer);
    thrust::exclusive_scan(triScanPtr, triScanPtr + m_deviceData->numCubes + 1, triScanPtr);
    m_deviceData->cubeFirstTriId.getDeviceObject(m_deviceData->numTriIds, m_deviceData->numCubes);

    thrust::device_ptr<int> vertScanPtr(m_deviceData->edgeVertexNr.buffer);
    thrust::exclusive_scan(vertScanPtr, vertScanPtr + 12 * m_deviceData->numCubes + 1, vertScanPtr);
    m_deviceData->edgeVertexNr.getDeviceObject(m_deviceData->numVertices, 12 * m_deviceData->numCubes);

    // 8. Create Mesh
    int* triIndices = nullptr;
    if (m_useBufferObjects) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_triIdsIbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_deviceData->numTriIds * sizeof(int), nullptr, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        cudaGraphicsMapResources(1, &m_cudaTriIboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&triIndices, nullptr, m_cudaTriIboResource);
    } else {
        m_deviceData->triIndices.resize(m_deviceData->numTriIds, false);
        triIndices = m_deviceData->triIndices.buffer;
    }

    kernel_createCubeTriangles<<<(m_deviceData->numCubes + 255) / 256, 256>>>(*m_deviceData, triIndices);
    cudaCheck(cudaGetLastError());
    if (m_useBufferObjects) cudaGraphicsUnmapResources(1, &m_cudaTriIboResource, 0);

    Vec3 *vertices = nullptr, *normals = nullptr;
    if (m_useBufferObjects) {
        glBindBuffer(GL_ARRAY_BUFFER, m_verticesVbo);
        glBufferData(GL_ARRAY_BUFFER, m_deviceData->numVertices * sizeof(Vec3), nullptr, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        cudaGraphicsMapResources(1, &m_cudaVerticesVboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&vertices, nullptr, m_cudaVerticesVboResource);

        glBindBuffer(GL_ARRAY_BUFFER, m_normalsVbo);
        glBufferData(GL_ARRAY_BUFFER, m_deviceData->numVertices * sizeof(Vec3), nullptr, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        cudaGraphicsMapResources(1, &m_cudaNormalsVboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&normals, nullptr, m_cudaNormalsVboResource);
    } else {
        m_deviceData->meshVertices.resize(m_deviceData->numVertices, false);
        m_deviceData->meshNormals.resize(m_deviceData->numVertices, false);
        vertices = m_deviceData->meshVertices.buffer;
        normals = m_deviceData->meshNormals.buffer;
    }

    kernel_createVerticesAndNormals<<<(m_deviceData->numCubes + 255) / 256, 256>>>(*m_deviceData, vertices, normals);
    cudaCheck(cudaGetLastError());
    if (m_useBufferObjects) {
        cudaGraphicsUnmapResources(1, &m_cudaVerticesVboResource, 0);
        cudaGraphicsUnmapResources(1, &m_cudaNormalsVboResource, 0);
    }

    return true;
}

// ... [Existing: readBackMesh, getNumVertices, getNumTriangles, free] ...

void MarchingCubesSurface::readBackMesh()
{
    if (!m_useBufferObjects)
    {
        m_deviceData->meshVertices.get(m_vertices);
        m_deviceData->meshNormals.get(m_normals);
        m_deviceData->triIndices.get(m_triIndices);
    }
}

int MarchingCubesSurface::getNumVertices() const { return m_initialized ? m_deviceData->numVertices : 0; }
int MarchingCubesSurface::getNumTriangles() const { return m_initialized ? m_deviceData->numTriIds / 3 : 0; }

void MarchingCubesSurface::free()
{
    if (m_cudaVerticesVboResource) { cudaGraphicsUnregisterResource(m_cudaVerticesVboResource); m_cudaVerticesVboResource = nullptr; }
    if (m_cudaNormalsVboResource) { cudaGraphicsUnregisterResource(m_cudaNormalsVboResource); m_cudaNormalsVboResource = nullptr; }
    if (m_cudaTriIboResource) { cudaGraphicsUnregisterResource(m_cudaTriIboResource); m_cudaTriIboResource = nullptr; }
    if (m_verticesVbo) { glDeleteBuffers(1, &m_verticesVbo); m_verticesVbo = 0; }
    if (m_normalsVbo) { glDeleteBuffers(1, &m_normalsVbo); m_normalsVbo = 0; }
    if (m_triIdsIbo) { glDeleteBuffers(1, &m_triIdsIbo); m_triIdsIbo = 0; }
    if (m_deviceData) m_deviceData->free();
    m_deviceData = nullptr; m_initialized = false;
}