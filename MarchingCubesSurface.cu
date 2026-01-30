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
        sortedParticleIds.free();
        particleCoords.free();
        hashCellFirst.free();
        hashCellLast.free();
        cubeOfParticle.free();
        cornerDensities.free();
        cornerNormals.free();
        cubeCoords.free();
        cubeEdgeLinks.free();
        cubeFirstTriId.free();
        edgeVertexNr.free();

        // New persistent buffers to avoid cudaMalloc overhead
        uniqueOccupiedCoords.free();
        expandedCoords.free();

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
    float densityThreshold;
    int numTriIds;
    int numVertices;

    DeviceBuffer<float> particlePos;
    DeviceBuffer<int> sortedParticleIds; 
    DeviceBuffer<uint64_t> particleCoords;
    DeviceBuffer<int> hashCellFirst;
    DeviceBuffer<int> hashCellLast;
    DeviceBuffer<int> cubeOfParticle;
    DeviceBuffer<float> cornerDensities;
    DeviceBuffer<Vec3> cornerNormals;
    DeviceBuffer<uint64_t> cubeCoords;
    DeviceBuffer<int> cubeEdgeLinks;
    DeviceBuffer<int> cubeFirstTriId;
    DeviceBuffer<int> edgeVertexNr;

    // Temporary workspace buffers
    DeviceBuffer<uint64_t> uniqueOccupiedCoords;
    DeviceBuffer<uint64_t> expandedCoords;

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

__global__ void kernel_fillHash(MarchingCubesSurfaceDeviceData data, int numParticles, const float* positions, int stride) 
{
    int pNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (pNr >= numParticles) return;
    
    const float* posPtr = positions + pNr * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);
    
    int xi, yi, zi;
    getPosCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);

    data.particleCoords[pNr] = packCoords(xi, yi, zi);
    data.sortedParticleIds[pNr] = pNr;
}

__global__ void kernel_expandActiveVoxels(const uint64_t* occupiedCoords, uint64_t* expandedCoords, int numOccupied) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numOccupied) return;

    uint64_t centerCoord = occupiedCoords[idx];
    int xi, yi, zi;
    unpackCoords(centerCoord, xi, yi, zi);

    // Expand to 3x3x3 neighborhood (27 cells)
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int outIdx = idx * 27 + ((z + 1) * 9 + (y + 1) * 3 + (x + 1));
                expandedCoords[outIdx] = packCoords(xi + x, yi + y, zi + z);
            }
        }
    }
}

__global__ void kernel_setBoundaries(MarchingCubesSurfaceDeviceData data, int numCubes) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= numCubes) return;

    uint64_t coord = data.cubeCoords[cubeNr];
    int xi, yi, zi;
    unpackCoords(coord, xi, yi, zi);
    int h = hashFunction(xi, yi, zi);

    // Standard hash grid start/end pointers for O(1) cube lookup
    if (cubeNr == 0) {
        data.hashCellFirst[h] = 0;
    } else {
        int prevXi, prevYi, prevZi;
        unpackCoords(data.cubeCoords[cubeNr - 1], prevXi, prevYi, prevZi);
        int prevH = hashFunction(prevXi, prevYi, prevZi);

        if (h != prevH) {
            data.hashCellFirst[h] = cubeNr;
            data.hashCellLast[prevH] = cubeNr;
        }
    }
    
    if (cubeNr == numCubes - 1) {
        data.hashCellLast[h] = numCubes;
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
                int first = data.hashCellFirst[hash];
                int last = data.hashCellLast[hash];

                for (int j = first; j < last; j++) {
                    if (data.cubeCoords[j] == packCoords(adjXi, adjYi, adjZi)) {
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

    uint64_t coord = data.cubeCoords[cubeNr];
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
        for (int j = data.hashCellFirst[h]; j < data.hashCellLast[h]; j++) {
            if (data.cubeCoords[j] == packCoords(adjXi, adjYi, adjZi)) {
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
    unpackCoords(data.cubeCoords[cubeNr], xi, yi, zi);
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

    m_deviceData->hashCellFirst.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->hashCellLast.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->particleCoords.resize(numParticles, false);
    m_deviceData->sortedParticleIds.resize(numParticles, false);

    // Pre-allocate temporary workspace to particle size to avoid per-frame malloc
    m_deviceData->uniqueOccupiedCoords.resize(numParticles, false); 
    m_deviceData->expandedCoords.resize(numParticles * 27, false);

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
        m_deviceData->particlePos.set(particlePositions, numParticles);
        particles = m_deviceData->particlePos.buffer;
    }

    // 1. Fill particle hash and sort
    kernel_fillHash<<<(numParticles + 255) / 256, 256>>>(*m_deviceData, numParticles, particles, stride);
    thrust::device_ptr<uint64_t> pCoordsPtr(m_deviceData->particleCoords.buffer);
    thrust::device_ptr<int> sortedIdsPtr(m_deviceData->sortedParticleIds.buffer);
    thrust::sort_by_key(pCoordsPtr, pCoordsPtr + numParticles, sortedIdsPtr);

    // 2. Extract unique occupied voxels
    m_deviceData->uniqueOccupiedCoords.resize(numParticles, false);
    thrust::device_ptr<uint64_t> uniqueOccupiedPtr(m_deviceData->uniqueOccupiedCoords.buffer);
    auto uniqueEnd = thrust::unique_copy(pCoordsPtr, pCoordsPtr + numParticles, uniqueOccupiedPtr);
    int numOccupied = uniqueEnd - uniqueOccupiedPtr;

    // 3. Expand to include the neighbor "shell" (Active Cubes)
    int totalCandidates = numOccupied * 27;
    m_deviceData->expandedCoords.resize(totalCandidates, false);
    kernel_expandActiveVoxels<<<(numOccupied + 255) / 256, 256>>>(m_deviceData->uniqueOccupiedCoords.buffer, m_deviceData->expandedCoords.buffer, numOccupied);

    // 4. Finalize unique active cube list
    thrust::device_ptr<uint64_t> expandedPtr(m_deviceData->expandedCoords.buffer);
    thrust::sort(expandedPtr, expandedPtr + totalCandidates);
    auto activeEnd = thrust::unique(expandedPtr, expandedPtr + totalCandidates);
    int numActiveCubes = activeEnd - expandedPtr;

    m_deviceData->numCubes = numActiveCubes;
    m_deviceData->cubeCoords.resize(numActiveCubes, false);
    cudaMemcpy(m_deviceData->cubeCoords.buffer, m_deviceData->expandedCoords.buffer, numActiveCubes * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    // 5. Setup Hash for Cube Lookup
    m_deviceData->hashCellFirst.setZero();
    m_deviceData->hashCellLast.setZero();
    kernel_setBoundaries<<<(numActiveCubes + 255) / 256, 256>>>(*m_deviceData, numActiveCubes);

    // 6. Accumulate Density
    m_deviceData->cornerDensities.resize(8 * numActiveCubes, false);
    m_deviceData->cornerDensities.setZero();
    m_deviceData->cornerNormals.resize(8 * numActiveCubes, false);
    m_deviceData->cornerNormals.setZero();
    kernel_addParticleDensities<<<(numParticles + 255) / 256, 256>>>(*m_deviceData, numParticles, particles, stride);

    // 7. Surface Generation
    m_deviceData->cubeEdgeLinks.resize(12 * numActiveCubes, false);
    m_deviceData->cubeEdgeLinks.setZero();
    m_deviceData->edgeVertexNr.resize(12 * numActiveCubes + 1, false);
    m_deviceData->edgeVertexNr.setZero();
    m_deviceData->cubeFirstTriId.resize(numActiveCubes + 1, false);
    m_deviceData->cubeFirstTriId.setZero();

    kernel_sumCornerDensitiesAndFindEdgeLinks<<<(numActiveCubes + 255) / 256, 256>>>(*m_deviceData);
    
    // Scan for triangle and vertex counts
    thrust::device_ptr<int> triScanPtr(m_deviceData->cubeFirstTriId.buffer);
    thrust::exclusive_scan(triScanPtr, triScanPtr + numActiveCubes + 1, triScanPtr);
    m_deviceData->cubeFirstTriId.getDeviceObject(m_deviceData->numTriIds, numActiveCubes);

    thrust::device_ptr<int> vertScanPtr(m_deviceData->edgeVertexNr.buffer);
    thrust::exclusive_scan(vertScanPtr, vertScanPtr + 12 * numActiveCubes + 1, vertScanPtr);
    m_deviceData->edgeVertexNr.getDeviceObject(m_deviceData->numVertices, 12 * numActiveCubes);

    // 8. Create Mesh
    int* triIndices = nullptr;
    if (m_useBufferObjects) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_triIdsIbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_deviceData->numTriIds * sizeof(int), nullptr, GL_STATIC_DRAW);
        cudaGraphicsMapResources(1, &m_cudaTriIboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&triIndices, nullptr, m_cudaTriIboResource);
    } else {
        m_deviceData->triIndices.resize(m_deviceData->numTriIds, false);
        triIndices = m_deviceData->triIndices.buffer;
    }

    kernel_createCubeTriangles<<<(numActiveCubes + 255) / 256, 256>>>(*m_deviceData, triIndices);
    if (m_useBufferObjects) cudaGraphicsUnmapResources(1, &m_cudaTriIboResource, 0);

    Vec3 *vertices = nullptr, *normals = nullptr;
    if (m_useBufferObjects) {
        glBindBuffer(GL_ARRAY_BUFFER, m_verticesVbo);
        glBufferData(GL_ARRAY_BUFFER, m_deviceData->numVertices * sizeof(Vec3), nullptr, GL_STATIC_DRAW);
        cudaGraphicsMapResources(1, &m_cudaVerticesVboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&vertices, nullptr, m_cudaVerticesVboResource);

        glBindBuffer(GL_ARRAY_BUFFER, m_normalsVbo);
        glBufferData(GL_ARRAY_BUFFER, m_deviceData->numVertices * sizeof(Vec3), nullptr, GL_STATIC_DRAW);
        cudaGraphicsMapResources(1, &m_cudaNormalsVboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&normals, nullptr, m_cudaNormalsVboResource);
    } else {
        m_deviceData->meshVertices.resize(m_deviceData->numVertices, false);
        m_deviceData->meshNormals.resize(m_deviceData->numVertices, false);
        vertices = m_deviceData->meshVertices.buffer;
        normals = m_deviceData->meshNormals.buffer;
    }

    kernel_createVerticesAndNormals<<<(numActiveCubes + 255) / 256, 256>>>(*m_deviceData, vertices, normals);
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
    if (m_verticesVbo) glDeleteBuffers(1, &m_verticesVbo);
    if (m_normalsVbo) glDeleteBuffers(1, &m_normalsVbo);
    if (m_triIdsIbo) glDeleteBuffers(1, &m_triIdsIbo);
    if (m_cudaVerticesVboResource) cudaGraphicsUnregisterResource(m_cudaVerticesVboResource);
    if (m_cudaNormalsVboResource) cudaGraphicsUnregisterResource(m_cudaNormalsVboResource);
    if (m_cudaTriIboResource) cudaGraphicsUnregisterResource(m_cudaTriIboResource);
    if (m_deviceData) m_deviceData->free();
    m_deviceData = nullptr; m_initialized = false;
}