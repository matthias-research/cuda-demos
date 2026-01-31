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
#include <thrust/execution_policy.h>
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
        particleIndices.free();
        cubes.free();
        hashFirstCube.free();
        hashLastCube.free();
        firstExpandedCube.free();
        neighborBits.free();
        cornerDensities.free();
        cornerNormals.free();
        cubeEdgeLinks.free();
        cubeFirstTriId.free();
        edgeVertexNr.free();
        meshVertices.free();
        meshNormals.free();
        triIndices.free();
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
    DeviceBuffer<int> particleIndices; // For Gather Indirection
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
    unsigned int uxi = (unsigned int)xi;
    unsigned int uyi = (unsigned int)yi;
    unsigned int uzi = (unsigned int)zi;
    return ((uxi * 92837111u) ^ (uyi * 689287499u) ^ (uzi * 283923481u)) % MARCHING_CUBES_HASH_SIZE;
}

__device__ inline void getPosCoords(const Vec3& pos, float invGridSpacing, float worldOrig, int& xi, int& yi, int& zi)
{
    xi = (int)floorf((pos.x - worldOrig) * invGridSpacing);
    yi = (int)floorf((pos.y - worldOrig) * invGridSpacing);
    zi = (int)floorf((pos.z - worldOrig) * invGridSpacing);
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
    if (pNr >= numParticles) return;
    
    const float* posPtr = positions + pNr * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);
    
    int xi, yi, zi;
    getPosCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);
    data.particleCubes[pNr] = packCoords(xi, yi, zi);
}

__global__ void kernel_setHashBoundaries(MarchingCubesSurfaceDeviceData data, int numItems, bool particleMode) 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= numItems) return;

    uint64_t* items = particleMode ? data.particleCubes.buffer : data.cubes.buffer;

    int xi, yi, zi;
    unpackCoords(items[i], xi, yi, zi);
    int h = hashFunction(xi, yi, zi);

    if (i == 0) {
        data.hashFirstCube[h] = 0;
    } else {
        int pXi, pYi, pZi;
        unpackCoords(items[i - 1], pXi, pYi, pZi);
        int prevH = hashFunction(pXi, pYi, pZi);

        if (h != prevH) {
            data.hashFirstCube[h] = i;
            data.hashLastCube[prevH] = i;
        }
    }
    
    if (i == numItems - 1) {
        data.hashLastCube[h] = numItems;
    }
}

__global__ void kernel_findExpandedShell(MarchingCubesSurfaceDeviceData data) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numParticles) return;

    if (cubeNr > 0 && data.particleCubes[cubeNr] == data.particleCubes[cubeNr - 1]) return;

    int cx, cy, cz;
    unpackCoords(data.particleCubes[cubeNr], cx, cy, cz);

    int numNeighbors = 0;
    unsigned int bits = 0;
    int neighborNr = 0;

    for (int xi = cx - 1; xi <= cx + 1; xi++) {
        for (int yi = cy - 1; yi <= cy + 1; yi++) {
            for (int zi = cz - 1; zi <= cz + 1; zi++) {
                uint64_t adjCoord = packCoords(xi, yi, zi);
                int h = hashFunction(xi, yi, zi);
                bool found = false;
                for (int j = data.hashFirstCube[h]; j < data.hashLastCube[h]; j++) {
                    if (data.particleCubes[j] == adjCoord) { found = true; break; }
                }
                if (found) { bits |= (1 << neighborNr); numNeighbors++; }
                neighborNr++;
            }
        }
    }
    data.firstExpandedCube[cubeNr] = numNeighbors;
    data.neighborBits[cubeNr] = bits;
}

__global__ void kernel_createExpandedShell(MarchingCubesSurfaceDeviceData data) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numParticles) return;
    if (cubeNr > 0 && data.particleCubes[cubeNr] == data.particleCubes[cubeNr - 1]) return;

    int cx, cy, cz;
    unpackCoords(data.particleCubes[cubeNr], cx, cy, cz);
    unsigned int bits = data.neighborBits[cubeNr];
    int pos = data.firstExpandedCube[cubeNr];
    int neighborNr = 0;

    for (int xi = cx - 1; xi <= cx + 1; xi++) {
        for (int yi = cy - 1; yi <= cy + 1; yi++) {
            for (int zi = cz - 1; zi <= cz + 1; zi++) {
                if (bits & (1 << neighborNr)) {
                    data.cubes[pos++] = packCoords(xi, yi, zi);
                }
                neighborNr++;
            }
        }
    }
}

// THE GATHER KERNEL (Atomic-Free)
__global__ void kernel_gatherParticleDensities(MarchingCubesSurfaceDeviceData data, const float* positions, int stride) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes) return;

    int cx, cy, cz;
    unpackCoords(data.cubes[cubeNr], cx, cy, cz);

    float h_rad = 1.5f * data.gridSpacing;
    float h2 = h_rad * h_rad;
    float kernelScale = 315.0f / (64.0f * 3.14159265f * h2 * h2 * h2 * h2 * h_rad) / data.restDensity;
    Vec3 cellOrig = Vec3(cx * data.gridSpacing, cy * data.gridSpacing, cz * data.gridSpacing) + data.worldOrig;

    for (int i = 0; i < 8; i++) {
        Vec3 cornerPos = cellOrig + Vec3((float)marchingCubeCorners[i][0], (float)marchingCubeCorners[i][1], (float)marchingCubeCorners[i][2]) * data.gridSpacing;
        float sumD = 0.0f;
        Vec3 sumN(0,0,0);

        // Check 3x3x3 cell neighborhood for particles
        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int h = hashFunction(cx + dx, cy + dy, cz + dz);
                    uint64_t targetCoord = packCoords(cx + dx, cy + dy, cz + dz);
                    for (int j = data.hashFirstCube[h]; j < data.hashLastCube[h]; j++) {
                        if (data.particleCubes[j] == targetCoord) {
                            int pIdx = data.particleIndices[j]; // INDIRECTION
                            const float* pPtr = positions + pIdx * stride;
                            Vec3 r = Vec3(pPtr[0], pPtr[1], pPtr[2]) - cornerPos;
                            float r2 = r.magnitudeSquared();
                            if (r2 < h2) {
                                float w = kernelScale * (h2 - r2) * (h2 - r2) * (h2 - r2);
                                sumD += w;
                                float rLen = sqrtf(r2);
                                if (rLen > 1e-8f) sumN += r * (-w / rLen);
                            }
                        }
                    }
                }
            }
        }
        data.cornerDensities[8 * cubeNr + i] = sumD;
        data.cornerNormals[8 * cubeNr + i] = sumN;
    }
}

// ... [Remaining Surface Generation Kernels: kernel_sumCornerDensitiesAndFindEdgeLinks, etc.] ...

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
        int ax = xi + (i % 3) - 1, ay = yi + ((i / 3) % 3) - 1, az = zi + ((i / 9) % 3) - 1;
        int h = hashFunction(ax, ay, az);
        uint64_t target = packCoords(ax, ay, az);
        int foundIdx = -1;
        for (int j = data.hashFirstCube[h]; j < data.hashLastCube[h]; j++) {
            if (data.cubes[j] == target) { foundIdx = j; break; }
        }
        adjCubes[i] = foundIdx;
    }

    int code = 0;
    for (int cornerNr = 0; cornerNr < 8; cornerNr++) {
        float density = data.cornerDensities[8 * cubeNr + cornerNr];
        Vec3 normal = data.cornerNormals[8 * cubeNr + cornerNr];

        for (int i = 0; i < 8; i++) {
            int adj = adjCubes[cornerAdjCellNr[cornerNr][i]];
            if (adj >= 0) {
                density += data.cornerDensities[8 * adj + cornerAdjCornerNr[cornerNr][i]];
                normal += data.cornerNormals[8 * adj + cornerAdjCornerNr[cornerNr][i]];
            }
        }
        data.cornerDensities[8 * cubeNr + cornerNr] = density;
        data.cornerNormals[8 * cubeNr + cornerNr] = normal;
        if (density > data.densityThreshold) code |= (1 << cornerNr);
    }

    data.cubeFirstTriId[cubeNr] = firstMarchingCubesId[code + 1] - firstMarchingCubesId[code];

    for (int edgeNr = 0; edgeNr < 12; edgeNr++) {
        bool smallest = true;
        for (int i = 0; i < 4; i++) {
            int adj = adjCubes[edgeAdjCellNr[edgeNr][i]];
            if (adj >= 0 && adj < cubeNr) smallest = false;
        }

        if (smallest) {
            for (int i = 0; i < 4; i++) {
                int adj = adjCubes[edgeAdjCellNr[edgeNr][i]];
                if (adj >= 0) data.cubeEdgeLinks[12 * adj + edgeAdjEdgeNr[edgeNr][i]] = 12 * cubeNr + edgeNr;
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

    int code = 0;
    for (int i = 0; i < 8; i++) if (data.cornerDensities[8 * cubeNr + i] > data.densityThreshold) code |= (1 << i);
    if (code == 0) return;

    int tableStart = firstMarchingCubesId[code];
    int outPos = data.cubeFirstTriId[cubeNr];
    for (int i = 0; i < (firstMarchingCubesId[code + 1] - tableStart); i++) {
        triIndices[outPos + i] = edgeVertexId[marchingCubesIds[tableStart + i]];
    }
}

__global__ void kernel_createVerticesAndNormals(MarchingCubesSurfaceDeviceData data, Vec3* vertices, Vec3* normals) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes) return;

    int xi, yi, zi;
    unpackCoords(data.cubes[cubeNr], xi, yi, zi);
    Vec3 cubeOrig = Vec3(xi * data.gridSpacing, yi * data.gridSpacing, zi * data.gridSpacing) + data.worldOrig;

    for (int edgeNr = 0; edgeNr < 12; edgeNr++) {
        if (data.cubeEdgeLinks[12 * cubeNr + edgeNr] >= 0) continue;

        int id0 = marchingCubeEdges[edgeNr][0], id1 = marchingCubeEdges[edgeNr][1];
        float d0 = data.cornerDensities[8 * cubeNr + id0], d1 = data.cornerDensities[8 * cubeNr + id1];

        if ((d0 <= data.densityThreshold && d1 > data.densityThreshold) || (d0 > data.densityThreshold && d1 <= data.densityThreshold)) {
            float t = (fabsf(d1 - d0) > 1e-6f) ? Clamp((data.densityThreshold - d0) / (d1 - d0), 0.0f, 1.0f) : 0.5f;
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
    m_deviceData->particleIndices.resize(numParticles, false);
    m_deviceData->hashFirstCube.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->hashLastCube.resize(MARCHING_CUBES_HASH_SIZE, false);

    if (useBufferObjects) {
        if (m_verticesVbo == 0) { glGenBuffers(1, &m_verticesVbo); cudaGraphicsGLRegisterBuffer(&m_cudaVerticesVboResource, m_verticesVbo, cudaGraphicsMapFlagsWriteDiscard); }
        if (m_normalsVbo == 0) { glGenBuffers(1, &m_normalsVbo); cudaGraphicsGLRegisterBuffer(&m_cudaNormalsVboResource, m_normalsVbo, cudaGraphicsMapFlagsWriteDiscard); }
        if (m_triIdsIbo == 0) { glGenBuffers(1, &m_triIdsIbo); cudaGraphicsGLRegisterBuffer(&m_cudaTriIboResource, m_triIdsIbo, cudaGraphicsMapFlagsWriteDiscard); }
    }
    m_useBufferObjects = useBufferObjects;
    m_initialized = true;
    return true;
}

bool MarchingCubesSurface::update(int numParticles, const float* particlePositions, int stride, bool onGpu)
{
    if (!m_deviceData || !m_initialized) return false;

    const float* particles = particlePositions;
    if (!onGpu) { m_deviceData->particlePos.set(particlePositions, numParticles * stride); particles = m_deviceData->particlePos.buffer; }

    // 1. Sort Particles via Indirection
    m_deviceData->particleIndices.resize(numParticles, false);
    thrust::sequence(thrust::device, m_deviceData->particleIndices.buffer, m_deviceData->particleIndices.buffer + numParticles);
    
    kernel_createParticleCubes<<<(numParticles + 255) / 256, 256>>>(*m_deviceData, numParticles, particles, stride);
    
    thrust::device_ptr<uint64_t> pCubesPtr(m_deviceData->particleCubes.buffer);
    thrust::device_ptr<int> pIndicesPtr(m_deviceData->particleIndices.buffer);
    thrust::sort_by_key(pCubesPtr, pCubesPtr + numParticles, pIndicesPtr);

    m_deviceData->hashFirstCube.setZero();
    m_deviceData->hashLastCube.setZero();
    kernel_setHashBoundaries<<<(numParticles + 255) / 256, 256>>>(*m_deviceData, numParticles, true);

    // 2. Expand Shell
    m_deviceData->firstExpandedCube.resize(numParticles + 1, false);
    m_deviceData->firstExpandedCube.setZero();
    m_deviceData->neighborBits.resize(numParticles, false);
    m_deviceData->neighborBits.setZero();

    kernel_findExpandedShell<<<(numParticles + 255) / 256, 256>>>(*m_deviceData);
    
    thrust::device_ptr<int> scanPtr(m_deviceData->firstExpandedCube.buffer);
    thrust::exclusive_scan(scanPtr, scanPtr + numParticles + 1, scanPtr);
    m_deviceData->firstExpandedCube.getDeviceObject(m_deviceData->numCubes, numParticles);

    m_deviceData->cubes.resize(m_deviceData->numCubes, false);
    kernel_createExpandedShell<<<(numParticles + 255) / 256, 256>>>(*m_deviceData);
    
    thrust::device_ptr<uint64_t> expPtr(m_deviceData->cubes.buffer);
    thrust::sort(expPtr, expPtr + m_deviceData->numCubes);
    auto expEnd = thrust::unique(expPtr, expPtr + m_deviceData->numCubes);
    m_deviceData->numCubes = expEnd - expPtr;

    m_deviceData->hashFirstCube.setZero();
    m_deviceData->hashLastCube.setZero();
    kernel_setHashBoundaries<<<(m_deviceData->numCubes + 255) / 256, 256>>>(*m_deviceData, m_deviceData->numCubes, false);

    // 3. Density - THE GATHER (Atomic-Free)
    m_deviceData->cornerDensities.resize(8 * m_deviceData->numCubes, false);
    m_deviceData->cornerNormals.resize(8 * m_deviceData->numCubes, false);
    kernel_gatherParticleDensities<<<(m_deviceData->numCubes + 255) / 256, 256>>>(*m_deviceData, particles, stride);

    // 4. Mesh Gen
    m_deviceData->cubeEdgeLinks.resize(12 * m_deviceData->numCubes, false);
    m_deviceData->edgeVertexNr.resize(12 * m_deviceData->numCubes + 1, false);
    m_deviceData->edgeVertexNr.setZero();
    m_deviceData->cubeFirstTriId.resize(m_deviceData->numCubes + 1, false);
    
    kernel_sumCornerDensitiesAndFindEdgeLinks<<<(m_deviceData->numCubes + 255) / 256, 256>>>(*m_deviceData);
    
    thrust::device_ptr<int> triScan(m_deviceData->cubeFirstTriId.buffer);
    thrust::exclusive_scan(triScan, triScan + m_deviceData->numCubes + 1, triScan);
    m_deviceData->cubeFirstTriId.getDeviceObject(m_deviceData->numTriIds, m_deviceData->numCubes);

    thrust::device_ptr<int> vertScan(m_deviceData->edgeVertexNr.buffer);
    thrust::exclusive_scan(vertScan, vertScan + 12 * m_deviceData->numCubes + 1, vertScan);
    m_deviceData->edgeVertexNr.getDeviceObject(m_deviceData->numVertices, 12 * m_deviceData->numCubes);

    // [Mesh Creation/VBO Logic - same as before but using numCubes]
    int* tris = nullptr;
    if (m_useBufferObjects) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_triIdsIbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_deviceData->numTriIds * sizeof(int), nullptr, GL_STATIC_DRAW);
        cudaGraphicsMapResources(1, &m_cudaTriIboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&tris, nullptr, m_cudaTriIboResource);
    } else { m_deviceData->triIndices.resize(m_deviceData->numTriIds, false); tris = m_deviceData->triIndices.buffer; }

    kernel_createCubeTriangles<<<(m_deviceData->numCubes + 255) / 256, 256>>>(*m_deviceData, tris);
    if (m_useBufferObjects) cudaGraphicsUnmapResources(1, &m_cudaTriIboResource, 0);

    Vec3 *v = nullptr, *n = nullptr;
    if (m_useBufferObjects) {
        glBindBuffer(GL_ARRAY_BUFFER, m_verticesVbo); glBufferData(GL_ARRAY_BUFFER, m_deviceData->numVertices * sizeof(Vec3), nullptr, GL_STATIC_DRAW);
        cudaGraphicsMapResources(1, &m_cudaVerticesVboResource, 0); cudaGraphicsResourceGetMappedPointer((void**)&v, nullptr, m_cudaVerticesVboResource);
        glBindBuffer(GL_ARRAY_BUFFER, m_normalsVbo); glBufferData(GL_ARRAY_BUFFER, m_deviceData->numVertices * sizeof(Vec3), nullptr, GL_STATIC_DRAW);
        cudaGraphicsMapResources(1, &m_cudaNormalsVboResource, 0); cudaGraphicsResourceGetMappedPointer((void**)&n, nullptr, m_cudaNormalsVboResource);
    } else {
        m_deviceData->meshVertices.resize(m_deviceData->numVertices, false); v = m_deviceData->meshVertices.buffer;
        m_deviceData->meshNormals.resize(m_deviceData->numVertices, false); n = m_deviceData->meshNormals.buffer;
    }

    kernel_createVerticesAndNormals<<<(m_deviceData->numCubes + 255) / 256, 256>>>(*m_deviceData, v, n);
    if (m_useBufferObjects) { cudaGraphicsUnmapResources(1, &m_cudaVerticesVboResource, 0); cudaGraphicsUnmapResources(1, &m_cudaNormalsVboResource, 0); }

    return true;
}

void MarchingCubesSurface::readBackMesh() {
    if (!m_useBufferObjects) { m_deviceData->meshVertices.get(m_vertices); m_deviceData->meshNormals.get(m_normals); m_deviceData->triIndices.get(m_triIndices); }
}

int MarchingCubesSurface::getNumVertices() const { return m_initialized ? m_deviceData->numVertices : 0; }
int MarchingCubesSurface::getNumTriangles() const { return m_initialized ? m_deviceData->numTriIds / 3 : 0; }

void MarchingCubesSurface::free() {
    if (m_cudaVerticesVboResource) cudaGraphicsUnregisterResource(m_cudaVerticesVboResource);
    if (m_cudaNormalsVboResource) cudaGraphicsUnregisterResource(m_cudaNormalsVboResource);
    if (m_cudaTriIboResource) cudaGraphicsUnregisterResource(m_cudaTriIboResource);
    if (m_verticesVbo) glDeleteBuffers(1, &m_verticesVbo);
    if (m_normalsVbo) glDeleteBuffers(1, &m_normalsVbo);
    if (m_triIdsIbo) glDeleteBuffers(1, &m_triIdsIbo);
    if (m_deviceData) m_deviceData->free();
    m_deviceData = nullptr; m_initialized = false;
}