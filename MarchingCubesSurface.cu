#define NOMINMAX
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "MarchingCubesSurface.h"
#include "CudaUtils.h"
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <vector>
#include "MarchingCubesTable.h"

static const int MARCHING_CUBES_HASH_SIZE = 37000111;  

struct MarchingCubesSurfaceDeviceData
{
    MarchingCubesSurfaceDeviceData()
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
        densityThreshold = 0.0f;
        numTriIds = 0;
        numVertices = 0;
    }

    // don't use destructor, cuda would call it in the kernel!

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

    // input, only used if CPU positions are provided
    DeviceBuffer<float> particlePos;

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
    DeviceBuffer<int> cubeFirstTriId;
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


__global__ void kernel_fillHash(MarchingCubesSurfaceDeviceData data, int numParticles, const float* positions, int stride) 
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


__global__ void kernel_setBoundaries(MarchingCubesSurfaceDeviceData data, int numParticles) 
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


__global__ void kernel_addParticleDensities(MarchingCubesSurfaceDeviceData data, int numParticles, const float* positions, int stride) 
{
    int posNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (posNr >= data.numParticles) 
        return;

    int id = data.sortedParticleIds[posNr];

    const float* posPtr = positions + id * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);

    int xi, yi, zi;
    getPosCoords(pos, data.invGridSpacing, data.worldOrig, xi, yi, zi);
    Vec3 worldOrig = Vec3(data.worldOrig, data.worldOrig, data.worldOrig);
    Vec3 cellOrig = Vec3(xi * data.gridSpacing, yi * data.gridSpacing, zi * data.gridSpacing) + worldOrig;

    int cubeNr = data.cubeOfParticle[posNr];

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
            float rLen = sqrtf(r2);
            if (rLen > 1e-8f) 
            {
                r *= (-w / rLen);
                atomicAdd(&data.cornerNormals[8 * cubeNr + i].x, r.x);
                atomicAdd(&data.cornerNormals[8 * cubeNr + i].y, r.y);
                atomicAdd(&data.cornerNormals[8 * cubeNr + i].z, r.z);
            }
        }   
    }
}


__global__ void kernel_sumCornerDensitiesAndFindEdgeLinks(MarchingCubesSurfaceDeviceData data) 
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
                adjCubeNr = data.cubeOfParticle[j];
                break;
            }
        }

        adjCubes[i] = adjCubeNr;
    }

    // sum densities

    int marchingCubesCode = 0;

    for (int cornerNr = 0; cornerNr < 8; cornerNr++)
    {
        float density = data.cornerDensities[8 * cubeNr + cornerNr];
        Vec3 normal = data.cornerNormals[8 * cubeNr + cornerNr];

        for (int i = 0; i < 8; i++)
        {
            int adjCubeNr = adjCubes[cornerAdjCellNr[cornerNr][i]];
            if (adjCubeNr < 0)
                continue;

            int adjCornerNr = cornerAdjCornerNr[cornerNr][i];

            float adjDensity = data.cornerDensities[8 * adjCubeNr + adjCornerNr];
            Vec3 adjCornerNormal = data.cornerNormals[8 * adjCubeNr + adjCornerNr];

            density += adjDensity;
            normal += adjCornerNormal;
        }

        data.cornerDensities[8 * cubeNr + cornerNr] = density;
        data.cornerNormals[8 * cubeNr + cornerNr] = normal;

        if (density > data.densityThreshold)
        {
            marchingCubesCode |= 1 << cornerNr;
        }
    }

    // compute the number of ids needed for this cube

    int firstIndex = firstMarchingCubesId[marchingCubesCode];
    int numIds = firstMarchingCubesId[marchingCubesCode + 1] - firstIndex;
    data.cubeFirstTriId[cubeNr] = numIds;

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

        if (!smallest) 
            continue;

        for (int i = 0; i < 4; i++)
        {
            int adjCubeNr = adjCubes[edgeAdjCellNr[edgeNr][i]];
            if (adjCubeNr < 0)
                continue;

            int adjEdgeNr = edgeAdjEdgeNr[edgeNr][i];

            data.cubeEdgeLinks[12 * adjCubeNr + adjEdgeNr] = 12 * cubeNr + edgeNr;
        }
        data.cubeEdgeLinks[12 * cubeNr + edgeNr] = -1;

        // will there be a vertex at this edge?

        int id0 = marchingCubeEdges[edgeNr][0];
        int id1 = marchingCubeEdges[edgeNr][1];
        float d0 = data.cornerDensities[8 * cubeNr + id0];
        float d1 = data.cornerDensities[8 * cubeNr + id1];

        if ((d0 <= data.densityThreshold && d1 > data.densityThreshold) || (d0 > data.densityThreshold && d1 <= data.densityThreshold))
        {
            data.edgeVertexNr[12 * cubeNr + edgeNr] = 1;
        }
    }
}


__global__ void kernel_createCubeTriangles(MarchingCubesSurfaceDeviceData data, int* triIndices) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes) 
        return;

    // each cube creates the triangles for its surface by marching through the iso-surface

    int edgeVertexId[12];

    for (int edgeNr = 0; edgeNr < 12; edgeNr++)
    {
        int link = data.cubeEdgeLinks[12 * cubeNr + edgeNr];
        int adjEdgeNr = link < 0 ? 12 * cubeNr + edgeNr : link;

        edgeVertexId[edgeNr] = data.edgeVertexNr[adjEdgeNr];
    }

    int marchingCubesCode = 0;
    for (int cornerNr = 0; cornerNr < 8; cornerNr++)
    {
        if (data.cornerDensities[8 * cubeNr + cornerNr] > data.densityThreshold)
        {
            marchingCubesCode |= 1 << cornerNr;
        }
    }

    if (marchingCubesCode == 0)
        return;

    int tableStart = firstMarchingCubesId[marchingCubesCode];
    int numIds = firstMarchingCubesId[marchingCubesCode + 1] - tableStart;
    int inPos = tableStart;
    int outPos = data.cubeFirstTriId[cubeNr];

    for (int i = 0; i < numIds; i++)
    {
        int localId = marchingCubesIds[inPos];
        int vertexId = edgeVertexId[localId];
        triIndices[outPos] = vertexId;
        inPos++;
        outPos++;
    }
}

__global__ void kernel_createVerticesAndNormals(MarchingCubesSurfaceDeviceData data, Vec3* vertices, Vec3* normals) 
{
    int cubeNr = threadIdx.x + blockIdx.x * blockDim.x;
    if (cubeNr >= data.numCubes)
        return;

    int xi, yi, zi;
    unpackCoords(data.cubeCoords[cubeNr], xi, yi, zi);
    Vec3 worldOrig = Vec3(data.worldOrig, data.worldOrig, data.worldOrig);
    Vec3 cubeOrig = Vec3(xi * data.gridSpacing, yi * data.gridSpacing, zi * data.gridSpacing) + worldOrig;

    for (int edgeNr = 0; edgeNr < 12; edgeNr++)
    {
        int link = data.cubeEdgeLinks[12 * cubeNr + edgeNr];
        if (link >= 0)
            continue;

        int id0 = marchingCubeEdges[edgeNr][0];
        int id1 = marchingCubeEdges[edgeNr][1];

        Vec3 p0 = cubeOrig + Vec3((float)marchingCubeCorners[id0][0], (float)marchingCubeCorners[id0][1], (float)marchingCubeCorners[id0][2]) * data.gridSpacing;
        Vec3 p1 = cubeOrig + Vec3((float)marchingCubeCorners[id1][0], (float)marchingCubeCorners[id1][1], (float)marchingCubeCorners[id1][2]) * data.gridSpacing;

        float d0 = data.cornerDensities[8 * cubeNr + id0];
        float d1 = data.cornerDensities[8 * cubeNr + id1];

        Vec3 n0 = data.cornerNormals[8 * cubeNr + id0];
        Vec3 n1 = data.cornerNormals[8 * cubeNr + id1];

        int id = data.edgeVertexNr[12 * cubeNr + edgeNr];

        if ((d0 <= data.densityThreshold && d1 > data.densityThreshold) || (d0 > data.densityThreshold && d1 <= data.densityThreshold))
        {
            float t = (d1 != d0) ? Clamp((data.densityThreshold - d0) / (d1 - d0), 0.0f, 1.0f) : 0.5f;

            Vec3 p = p0 + t * (p1 - p0);
            Vec3 n = n0 + t * (n1 - n0);
            n.normalize();

            vertices[id] = p;
            normals[id] = n;
        }
    }
}



// Host functions -------------------------------

MarchingCubesSurface::MarchingCubesSurface()
{
}

MarchingCubesSurface::~MarchingCubesSurface()
{
    free();
}

bool MarchingCubesSurface::initialize(int numParticles, float gridSpacing, bool useBufferObjects)
{
    m_vertices.clear();
    m_normals.clear();
    m_triIndices.clear();

    m_initialized = false;

    if (!m_deviceData || numParticles == 0)
    {
        m_deviceData = std::make_shared<MarchingCubesSurfaceDeviceData>();
        return false;
    }

    m_deviceData->gridSpacing = gridSpacing;
    m_deviceData->invGridSpacing = 1.0f / gridSpacing;
    m_deviceData->worldOrig = -1000.0f;
    m_deviceData->numParticles = numParticles;

    m_deviceData->hashCellFirst.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->hashCellLast.resize(MARCHING_CUBES_HASH_SIZE, false);
    m_deviceData->particleCoords.resize(numParticles, false);
    m_deviceData->sortedParticleIds.resize(numParticles, false);
    m_deviceData->cubeOfParticle.resize(numParticles + 1, false);

    if (useBufferObjects)
    {
        if (m_verticesVbo == 0)
        {
            glGenBuffers(1, &m_verticesVbo);
            cudaGraphicsGLRegisterBuffer(&m_cudaVerticesVboResource, m_verticesVbo, cudaGraphicsMapFlagsWriteDiscard);
        }
        if (m_normalsVbo == 0)
        {
            glGenBuffers(1, &m_normalsVbo);
            cudaGraphicsGLRegisterBuffer(&m_cudaNormalsVboResource, m_normalsVbo, cudaGraphicsMapFlagsWriteDiscard);
        }
        if (m_triIdsIbo == 0)
        {
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
    if (!m_deviceData || !m_initialized)
    {
        return false;
    }

    if (m_deviceData->numParticles != numParticles)
    {
        return false;
    }

    const float* particles = particlePositions;

    if (!onGpu)
    {
        m_deviceData->particlePos.set(particlePositions, numParticles);
        particles = m_deviceData->particlePos.buffer;
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
    thrust::exclusive_scan(cubeOfParticle, cubeOfParticle + numParticles + 1, cubeOfParticle);

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
    m_deviceData->cubeEdgeLinks.setZero();
    m_deviceData->edgeVertexNr.resize(12 * m_deviceData->numCubes + 1, false);
    m_deviceData->edgeVertexNr.setZero();
    m_deviceData->cubeFirstTriId.resize(m_deviceData->numCubes + 1, false);
    m_deviceData->cubeFirstTriId.setZero();

    kernel_sumCornerDensitiesAndFindEdgeLinks<<<m_deviceData->numCubes / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData);
    cudaCheck(cudaGetLastError());

    cudaDeviceSynchronize();

    thrust::device_ptr<int> cubeFirstTriId(m_deviceData->cubeFirstTriId.buffer);
    thrust::exclusive_scan(cubeFirstTriId, cubeFirstTriId + m_deviceData->numCubes + 1, cubeFirstTriId);
    m_deviceData->cubeFirstTriId.getDeviceObject(m_deviceData->numTriIds, m_deviceData->numCubes);

    // create vertex slots

    thrust::device_ptr<int> edgeVertexNr(m_deviceData->edgeVertexNr.buffer);
    thrust::exclusive_scan(edgeVertexNr, edgeVertexNr + 12 * m_deviceData->numCubes + 1, edgeVertexNr);

    m_deviceData->edgeVertexNr.getDeviceObject(m_deviceData->numVertices, 12 * m_deviceData->numCubes);

    // create triangles

    int* triIndices = nullptr;

    if (m_useBufferObjects)
    {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_triIdsIbo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_deviceData->numTriIds * sizeof(int), nullptr, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        cudaGraphicsMapResources(1, &m_cudaTriIboResource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&triIndices, nullptr, m_cudaTriIboResource);
    }
    else
    {
        m_deviceData->triIndices.resize(m_deviceData->numTriIds, false);
        m_deviceData->triIndices.setZero();
        triIndices = m_deviceData->triIndices.buffer;
    }

    kernel_createCubeTriangles<<<m_deviceData->numCubes / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData, triIndices);
    cudaCheck(cudaGetLastError());

    if (m_useBufferObjects)
    {
        cudaGraphicsUnmapResources(1, &m_cudaTriIboResource, 0);
    }

    // create vertices

    Vec3* vertices = nullptr;
    Vec3* normals = nullptr;

    if (m_useBufferObjects)
    {
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
    }
    else
    {
        m_deviceData->meshVertices.resize(m_deviceData->numVertices, false);
        m_deviceData->meshVertices.setZero();
        vertices = m_deviceData->meshVertices.buffer;

        m_deviceData->meshNormals.resize(m_deviceData->numVertices, false);
        m_deviceData->meshNormals.setZero();
        normals = m_deviceData->meshNormals.buffer;
    }

    kernel_createVerticesAndNormals<<<m_deviceData->numCubes / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*m_deviceData, vertices, normals);
    cudaCheck(cudaGetLastError());

    if (m_useBufferObjects)
    {
        cudaGraphicsUnmapResources(1, &m_cudaVerticesVboResource, 0);
        cudaGraphicsUnmapResources(1, &m_cudaNormalsVboResource, 0);
    }

    return true;
}

void MarchingCubesSurface::readBackMesh()
{
    if (m_useBufferObjects) // the data is returned via VBOs
    {
        m_vertices.clear();
        m_normals.clear();
        m_triIndices.clear();
    }
    else
    {
        m_deviceData->meshVertices.get(m_vertices);
        m_deviceData->meshNormals.get(m_normals);
        m_deviceData->triIndices.get(m_triIndices);
    }
}

int MarchingCubesSurface::getNumVertices() const
{
    return m_initialized ? m_deviceData->numVertices : 0;
}

int MarchingCubesSurface::getNumTriangles() const
{
    return m_initialized ? m_deviceData->numTriIds / 3 : 0;
}


void MarchingCubesSurface::free()
{
    if (m_verticesVbo)
    {
        glDeleteBuffers(1, &m_verticesVbo);
    }
    if (m_normalsVbo)
    {
        glDeleteBuffers(1, &m_normalsVbo);
    }
    if (m_triIdsIbo)
    {
        glDeleteBuffers(1, &m_triIdsIbo);
    }
    if (m_cudaVerticesVboResource)
    {
        cudaGraphicsUnregisterResource(m_cudaVerticesVboResource);
    }
    if (m_cudaNormalsVboResource)
    {
        cudaGraphicsUnregisterResource(m_cudaNormalsVboResource);
    }
    if (m_cudaTriIboResource)
    {
        cudaGraphicsUnregisterResource(m_cudaTriIboResource);
    }
    if (m_deviceData)
    {
        m_deviceData->free();
    }
    m_deviceData = nullptr;
    m_initialized = false;
}