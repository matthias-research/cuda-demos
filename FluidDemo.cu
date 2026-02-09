#include "FluidDemo.h"
#include "CudaUtils.h"
#include "CudaMeshes.h"
#include "CudaHash.h"
#include "MarchingCubesSurface.h"
#include "BVH.h"
#include "Scene.h"
#include "Mesh.h"
#include "Geometry.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>
#include <vector>

static const int VBO_STRIDE = 8;  // pos(3), color(3), lifetime(1), pad(1)
static const int VBO_POS = 0;
static const int VBO_COLOR = 3;
static const int VBO_LIFETIME = 6;

// Device data structure - all simulation state on GPU
struct FluidDeviceData {
    void free() {
        numParticles = 0;
        maxParticles = 0;
        vel.free();
        prevPos.free();
        posCorr.free();
        density.free();
        raycastHit.free();
        intBuffer.free();
        intBuffer2.free();
        intBuffer3.free();
        sourceParticlePos.free();
        sourceParticleVel.free();
    }

    int numParticles = 0;
    int maxParticles = 0;
    float particleRadius = 0.2f;
    float kernelRadius = 0.6f;     // SPH kernel support radius (h)
    float poly6Scale = 0.0f;       // Precomputed: 315 / (64 * π * h^9)
    float restDensity = 1.0f;      // Target density for uniform particle spacing
    float gridSpacing = 0.5f;
    float worldOrig = -100.0f;

    DeviceBuffer<Vec3> vel;
    DeviceBuffer<Vec3> prevPos;
    DeviceBuffer<Vec3> posCorr;
    DeviceBuffer<float> density;   // Per-particle density estimate
    DeviceBuffer<float> raycastHit;
    DeviceBuffer<int> intBuffer;
    DeviceBuffer<int> intBuffer2;
    DeviceBuffer<int> intBuffer3;
    DeviceBuffer<Vec3> sourceParticlePos;
    DeviceBuffer<Vec3> sourceParticleVel;

    float* vboData = nullptr;

    OBB sourceBounds = OBB(Transform(Identity), Vec3(10.0f, 20.0f, 10.0f));
    float sourceSpeed = 10.0f;
    float sourceDensity = 1.0f;
    OBB sinkBounds = OBB(Transform(Identity), Vec3(Zero));
};


// Integrate velocities and positions
__global__ void kernel_integrate(FluidDeviceData data, float dt, Vec3 gravity, float maxVelocity) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    data.prevPos[idx] = pos;

    Vec3 vel = data.vel[idx];
    vel += gravity * dt;

    float v = vel.length();
    if (v > maxVelocity)
        vel *= maxVelocity / v;

    data.vel[idx] = vel;
    pos += vel * dt;

    particleData[0] = pos.x;
    particleData[1] = pos.y;
    particleData[2] = pos.z;
}


// Poly6 kernel: W(r,h) = poly6Scale * (h² - r²)³ for r ≤ h
__device__ inline float poly6Kernel(float r2, float h2, float poly6Scale) {
    if (r2 >= h2) return 0.0f;
    float diff = h2 - r2;
    return poly6Scale * diff * diff * diff;
}


// Compute density for each particle using SPH and update color for visualization
__global__ void kernel_computeDensity(FluidDeviceData data, HashDeviceData hashData) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    float h = data.kernelRadius;
    float h2 = h * h;
    float poly6Scale = data.poly6Scale;

    // Self-contribution: r=0, so (h² - 0)³ = h⁶
    float density = poly6Scale * h2 * h2 * h2;

    int xi = floorf((pos.x - data.worldOrig) / data.gridSpacing);
    int yi = floorf((pos.y - data.worldOrig) / data.gridSpacing);
    int zi = floorf((pos.z - data.worldOrig) / data.gridSpacing);

    // Sum contributions from neighbors
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                unsigned int hash = hashFunction(xi + dx, yi + dy, zi + dz);
                int first = hashData.hashCellFirst[hash];
                int last = hashData.hashCellLast[hash];

                for (int i = first; i < last; i++) {
                    int otherIdx = hashData.hashIds[i];
                    if (otherIdx == idx) continue;

                    float* otherData = data.vboData + otherIdx * VBO_STRIDE;
                    Vec3 otherPos(otherData[0], otherData[1], otherData[2]);

                    Vec3 delta = pos - otherPos;
                    float r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

                    density += poly6Kernel(r2, h2, poly6Scale);
                }
            }
        }
    }

    data.density[idx] = density;

    // Update color based on density ratio for visualization
    float ratio = density / data.restDensity;
    
    // Map ratio to color: blue (< 1) -> white (1) -> red (> 1)
    float r, g, b;
    if (ratio < 1.0f) {
        // Under-dense: blue to white
        float t = ratio;  // 0 to 1
        r = t;
        g = t;
        b = 1.0f;
    } else {
        // Over-dense: white to red
        float t = fminf(ratio - 1.0f, 1.0f);  // 0 to 1
        r = 1.0f;
        g = 1.0f - t;
        b = 1.0f - t;
    }
    
    particleData[3] = r;
    particleData[4] = g;
    particleData[5] = b;
}


// Handle particle-particle collision - just push apart
__device__ inline void handleParticleCollision(
    int idx, int otherIdx,
    const Vec3& pos, const Vec3& otherPos,
    float radius,
    FluidDeviceData& data)
{
    if (idx >= otherIdx) return;

    Vec3 delta = otherPos - pos;
    float dist = delta.magnitude();
    float minDist = radius * 2.0f;

    if (dist < minDist && dist > 0.001f) {
        Vec3 normal = delta / dist;
        float overlap = minDist - dist;
        Vec3 separation = normal * (overlap * 0.5f);

        atomicAdd(&data.posCorr[idx].x, -separation.x);
        atomicAdd(&data.posCorr[idx].y, -separation.y);
        atomicAdd(&data.posCorr[idx].z, -separation.z);

        atomicAdd(&data.posCorr[otherIdx].x, separation.x);
        atomicAdd(&data.posCorr[otherIdx].y, separation.y);
        atomicAdd(&data.posCorr[otherIdx].z, separation.z);
    }
}


// Particle collision using spatial hash
__global__ void kernel_particleCollision_hash(FluidDeviceData data, HashDeviceData hashData) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);
    float radius = data.particleRadius;

    int xi = floorf((pos.x - data.worldOrig) / data.gridSpacing);
    int yi = floorf((pos.y - data.worldOrig) / data.gridSpacing);
    int zi = floorf((pos.z - data.worldOrig) / data.gridSpacing);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int cellX = xi + dx;
                int cellY = yi + dy;
                int cellZ = zi + dz;

                unsigned int h = hashFunction(cellX, cellY, cellZ);
                int first = hashData.hashCellFirst[h];
                int last = hashData.hashCellLast[h];

                for (int i = first; i < last; i++) {
                    int otherIdx = hashData.hashIds[i];
                    float* otherData = data.vboData + otherIdx * VBO_STRIDE;
                    Vec3 otherPos(otherData[0], otherData[1], otherData[2]);

                    handleParticleCollision(idx, otherIdx, pos, otherPos, radius, data);
                }
            }
        }
    }
}


// Wall collision - just push position outside
__global__ void kernel_wallCollision(FluidDeviceData data, Bounds3 sceneBounds, float radius) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    // Push outside walls
    if (pos.x - radius < sceneBounds.minimum.x) pos.x = sceneBounds.minimum.x + radius;
    if (pos.x + radius > sceneBounds.maximum.x) pos.x = sceneBounds.maximum.x - radius;
    if (pos.y - radius < sceneBounds.minimum.y) pos.y = sceneBounds.minimum.y + radius;
    if (pos.z - radius < sceneBounds.minimum.z) pos.z = sceneBounds.minimum.z + radius;
    if (pos.z + radius > sceneBounds.maximum.z) pos.z = sceneBounds.maximum.z - radius;

    particleData[0] = pos.x;
    particleData[1] = pos.y;
    particleData[2] = pos.z;
}


// Particle-mesh collision - push particle away from closest triangle
__device__ void kernel_particleMeshCollision(
    FluidDeviceData& data,
    const MeshesDeviceData& meshData,
    int particleIdx, int meshIdx,
    float searchRadius, float radius)
{
    float* particleData = data.vboData + particleIdx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    Bounds3 queryBounds(pos - Vec3(searchRadius, searchRadius, searchRadius),
                        pos + Vec3(searchRadius, searchRadius, searchRadius));

    int rootNode = meshData.trianglesBvh.mRootNodes[meshIdx];
    if (rootNode < 0) return;

    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;

    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        PackedNodeHalf lower = meshData.trianglesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = meshData.trianglesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf node
                int triIdx = leftIndex;

                int i0 = meshData.triIds.buffer[triIdx * 3 + 0];
                int i1 = meshData.triIds.buffer[triIdx * 3 + 1];
                int i2 = meshData.triIds.buffer[triIdx * 3 + 2];

                Vec3 v0 = meshData.vertices.buffer[i0];
                Vec3 v1 = meshData.vertices.buffer[i1];
                Vec3 v2 = meshData.vertices.buffer[i2];

                Vec3 baryCoords = getClosestPointOnTriangle(pos, v0, v1, v2);
                Vec3 closestPoint = v0 * baryCoords.x + v1 * baryCoords.y + v2 * baryCoords.z;

                Vec3 toParticle = pos - closestPoint;
                float dist = toParticle.magnitude();

                if (dist < radius) {
                    Vec3 normal = toParticle / dist;
                    pos = closestPoint + normal * radius;
                    if ((triIdx % 2) == 0)
                        pos += normal * 1.0f * radius;
                }
            }
            else {  // Internal node
                if (stackCount < 63) {
                    stack[stackCount++] = leftIndex;
                    stack[stackCount++] = rightIndex;
                }
            }
        }
    }

    particleData[0] = pos.x;
    particleData[1] = pos.y;
    particleData[2] = pos.z;
}


// Traverse mesh BVH for particle-mesh collision
__global__ void kernel_particleMeshesCollision(
    FluidDeviceData data,
    const MeshesDeviceData meshData,
    float searchRadius, float radius)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;
    if (meshData.numMeshTriangles == 0) return;

    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    Bounds3 queryBounds(pos - Vec3(searchRadius, searchRadius, searchRadius),
                        pos + Vec3(searchRadius, searchRadius, searchRadius));

    int rootNode = meshData.meshesBvh.mRootNodes[0];
    if (rootNode < 0) return;

    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;

    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        PackedNodeHalf lower = meshData.meshesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = meshData.meshesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf - mesh
                kernel_particleMeshCollision(data, meshData, idx, leftIndex, searchRadius, radius);
            }
            else {
                if (stackCount < 63) {
                    stack[stackCount++] = leftIndex;
                    stack[stackCount++] = rightIndex;
                }
            }
        }
    }
}


// Apply position corrections
__global__ void kernel_applyCorrections(FluidDeviceData data, float relaxation) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    pos += data.posCorr[idx] * relaxation;

    particleData[0] = pos.x;
    particleData[1] = pos.y;
    particleData[2] = pos.z;
}


// Derive velocity from position change (PBD)
__global__ void kernel_deriveVelocity(FluidDeviceData data, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    data.vel[idx] = (pos - data.prevPos[idx]) / dt;
}


// Initialize particles (only initializes particles where idx < numParticles)
__global__ void kernel_initParticles(
    FluidDeviceData data,
    Bounds3 spawnBounds,
    int particlesPerLayer, int particlesPerRow, int particlesPerCol,
    float gridSpacing, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;  // Only initialize active particles, not all buffer slots

    // Fast pseudo-random
    unsigned int hash = idx + seed;
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash + (hash << 3);
    hash = hash ^ (hash >> 4);
    hash = hash * 0x27d4eb2d;
    hash = hash ^ (hash >> 15);
    #define FRAND() (hash = hash * 1664525u + 1013904223u, float(hash) / float(0xFFFFFFFFu))

    float* particleData = data.vboData + idx * VBO_STRIDE;

    // Grid-based position
    int layer = idx / particlesPerLayer;
    int inLayer = idx % particlesPerLayer;
    int row = inLayer / particlesPerRow;
    int col = inLayer % particlesPerRow;

    particleData[0] = spawnBounds.minimum.x + gridSpacing + col * gridSpacing;
    particleData[2] = spawnBounds.minimum.z + gridSpacing + row * gridSpacing;
    particleData[1] = spawnBounds.minimum.y + gridSpacing + layer * gridSpacing;

    // Water-like blue color with slight variation
    float colorVar = FRAND() * 0.2f;
    particleData[3] = 0.2f + colorVar * 0.3f;  // R
    particleData[4] = 0.5f + colorVar;          // G
    particleData[5] = 0.9f + colorVar * 0.1f;   // B

    // Lifetime = 1.0 (alive)
    particleData[6] = 1.0f;

    // Padding
    particleData[7] = 0.0f;

    // Random initial velocity
    data.vel[idx].x = -1.0f + FRAND() * 2.0f;
    data.vel[idx].y = -1.0f + FRAND() * 2.0f;
    data.vel[idx].z = -1.0f + FRAND() * 2.0f;

    #undef FRAND
}

__global__ void kernel_markParticlesForRemoval(FluidDeviceData data) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);
    bool remove = false;

    // test lifetime
    if (particleData[VBO_LIFETIME] == 0.0f)
        remove = true;

    // test sink
    if (!data.sinkBounds.halfExtents.isZero()) 
    {
        Vec3 localPos = data.sinkBounds.trans.transformInv(pos);
        Vec3 halfExtents = data.sinkBounds.halfExtents;
        bool isInside = (localPos.x < -halfExtents.x || localPos.x > halfExtents.x ||
            localPos.y < -halfExtents.y || localPos.y > halfExtents.y ||
            localPos.z < -halfExtents.z || localPos.z > halfExtents.z);

        if (isInside)
            remove = true;
    }

    if (remove)
        data.intBuffer[idx] = 1;
    else
        data.intBuffer2[idx] = 1;
}

__global__ void kernel_listFullSlots(FluidDeviceData data, int numFullSlots) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    if (data.intBuffer2[idx + 1] > data.intBuffer2[idx]) // a full slot
    {
        int slotNumber = data.intBuffer2[idx];
        data.intBuffer3[numFullSlots - slotNumber - 1] = idx; // reverse order, consume from the end
    }
}

__global__ void kernel_fillEmptySlots(FluidDeviceData data, int numFullSlots) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;

    if (data.intBuffer2[idx + 1] > data.intBuffer2[idx]) // full slot
        return;

    int emptySlotNumber = data.intBuffer[idx];
    int fullSlotNumber = data.intBuffer3[emptySlotNumber];

    if (fullSlotNumber >= numFullSlots)
        return;

    // copy from source to empty slot
    float* emptyParticleData = data.vboData + emptySlotNumber * VBO_STRIDE;
    float* fullParticleData = data.vboData + fullSlotNumber * VBO_STRIDE;
    for (int i = 0; i < VBO_STRIDE; i++)
        emptyParticleData[i] = fullParticleData[i];

    data.vel[emptySlotNumber] = data.vel[fullSlotNumber];
}


__global__ void kernel_addSourceParticles(FluidDeviceData data, int numSourceParticles, Vec3 color, float lifetime) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= numSourceParticles) 
        return;

    int sourceIdx = idx;
    int destIdx = data.numParticles + idx;
    
    float* particleData = data.vboData + destIdx * VBO_STRIDE;
    particleData[VBO_POS] = data.sourceParticlePos[sourceIdx].x;
    particleData[VBO_POS + 1] = data.sourceParticlePos[sourceIdx].y;
    particleData[VBO_POS + 2] = data.sourceParticlePos[sourceIdx].z;
    particleData[VBO_COLOR] = color.x;
    particleData[VBO_COLOR + 1] = color.y;
    particleData[VBO_COLOR + 2] = color.z;
    particleData[VBO_LIFETIME] = lifetime;
    data.vel[destIdx] = data.sourceParticleVel[sourceIdx];
}


//-----------------------------------------------------------------------------
// Host functions

bool FluidDemo::cudaRaycast(const Ray& ray, float& minT) {
    minT = MaxFloat;
    if (!meshes) return false;

    deviceData->raycastHit.resize(1);
    meshes->rayCast(1, nullptr, ray, deviceData->raycastHit.buffer, 1);
    deviceData->raycastHit.getDeviceObject(minT, 0);
    return minT < MaxFloat;
}


void FluidDemo::initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, Scene* scene) {
    if (!deviceData)
        deviceData = std::make_shared<FluidDeviceData>();
    if (!meshes)
        meshes = std::make_shared<CudaMeshes>();
    if (!hash)
        hash = std::make_shared<CudaHash>();

    meshes->initialize(scene);
    hash->initialize();

    deviceData->maxParticles = demoDesc.numMaxParticles;
    deviceData->numParticles = demoDesc.numParticles;  // Can be 0 to start empty
    deviceData->particleRadius = demoDesc.particleRadius;
    deviceData->kernelRadius = demoDesc.kernelRadius;
    deviceData->worldOrig = -100.0f;
    deviceData->gridSpacing = demoDesc.kernelRadius;  // Grid spacing = kernel radius for neighbor search
    
    // Copy source and sink bounds from descriptor
    deviceData->sourceBounds = demoDesc.sourceBounds;
    deviceData->sinkBounds = demoDesc.sinkBounds;
    deviceData->sourceSpeed = demoDesc.sourceSpeed;
    deviceData->sourceDensity = demoDesc.sourceDensity;

    // Precompute Poly6 kernel scale: 315 / (64 * π * h^9)
    float h = demoDesc.kernelRadius;
    float h2 = h * h;
    float h3 = h2 * h;
    float h6 = h3 * h3;
    float h9 = h6 * h3;
    float poly6Scale = 315.0f / (64.0f * 3.14159265f * h9);
    deviceData->poly6Scale = poly6Scale;

    // Compute rest density analytically based on spawn spacing
    // Particles spawn on a cubic grid with spacing s = 2 * particleRadius
    float s = 2.0f * demoDesc.particleRadius;
    float s2 = s * s;
    
    // Self contribution: W(0) = poly6Scale * h^6
    float restDensity = poly6Scale * h6;
    
    // 6 face neighbors at distance s
    if (s < h) {
        float diff = h2 - s2;
        restDensity += 6.0f * poly6Scale * diff * diff * diff;
    }
    
    // 12 edge neighbors at distance s*sqrt(2)
    float s2_edge = 2.0f * s2;  // (s*sqrt(2))^2 = 2*s^2
    if (s2_edge < h2) {
        float diff = h2 - s2_edge;
        restDensity += 12.0f * poly6Scale * diff * diff * diff;
    }
    
    // 8 corner neighbors at distance s*sqrt(3)
    float s2_corner = 3.0f * s2;  // (s*sqrt(3))^2 = 3*s^2
    if (s2_corner < h2) {
        float diff = h2 - s2_corner;
        restDensity += 8.0f * poly6Scale * diff * diff * diff;
    }
    
    deviceData->restDensity = restDensity;
    printf("FluidDemo: kernelRadius=%.3f, spawnSpacing=%.3f, restDensity=%.6f\n", h, s, restDensity);

    // Set marching cubes density threshold (isosurface at ~50% of rest density)
    if (marchingCubesSurface) {
        float threshold = restDensity * 0.5f;
        marchingCubesSurface->setDensityThreshold(threshold);
        printf("FluidDemo: Marching cubes threshold=%.6f\n", threshold);
    }

    // Initialize all buffers to maxParticles size (fixed buffer)
    deviceData->vel.resize(demoDesc.numMaxParticles, false);
    deviceData->prevPos.resize(demoDesc.numMaxParticles, false);
    deviceData->posCorr.resize(demoDesc.numMaxParticles, false);
    deviceData->density.resize(demoDesc.numMaxParticles, false);

    deviceData->vel.setZero();

    // CUDA-OpenGL interop
    cudaGraphicsGLRegisterBuffer(vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, *vboResource);

    deviceData->vboData = d_vboData;

    // Calculate grid layout (only used if numParticles > 0)
    float gridSpacing = 2.0f * demoDesc.particleRadius;
    Vec3 spawnSize = demoDesc.spawnBounds.maximum - demoDesc.spawnBounds.minimum;
    float usableWidth = spawnSize.x - 2.0f * gridSpacing;
    float usableDepth = spawnSize.z - 2.0f * gridSpacing;
    int particlesPerRow = (int)(usableWidth / gridSpacing);
    if (particlesPerRow < 1) particlesPerRow = 1;
    int particlesPerCol = (int)(usableDepth / gridSpacing);
    if (particlesPerCol < 1) particlesPerCol = 1;
    int particlesPerLayer = particlesPerRow * particlesPerCol;

    // Only initialize particles if numParticles > 0
    if (deviceData->numParticles > 0) {
        int numBlocks = (deviceData->numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        printf("FluidDemo: Initializing %d particles (max capacity: %d)\n", deviceData->numParticles, deviceData->maxParticles);

        kernel_initParticles<<<numBlocks, THREADS_PER_BLOCK>>>(
            *deviceData, demoDesc.spawnBounds,
            particlesPerLayer, particlesPerRow, particlesPerCol,
            gridSpacing, (unsigned long)time(nullptr));
    } else {
        printf("FluidDemo: Starting with 0 particles (max capacity: %d)\n", deviceData->maxParticles);
    }

    // Generate initial marching cubes surface so it's visible before simulation starts
    if (marchingCubesSurface) {
        marchingCubesSurface->update(deviceData->numParticles, deviceData->vboData, VBO_STRIDE, true);
    }

    cudaGraphicsUnmapResources(1, vboResource, 0);
}


void FluidDemo::removeParticles() 
{
    if (deviceData->numParticles == 0)
        return;

    deviceData->intBuffer.resize(deviceData->numParticles + 1, false);
    deviceData->intBuffer.setZero();

    deviceData->intBuffer2.resize(deviceData->numParticles + 1, false);
    deviceData->intBuffer2.setZero();

    kernel_markParticlesForRemoval << <(deviceData->numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (*deviceData);
    cudaCheck(cudaGetLastError());

    thrust::device_ptr<int> firstEmptySlot(deviceData->intBuffer.buffer);
    thrust::exclusive_scan(firstEmptySlot, firstEmptySlot + deviceData->numParticles + 1, firstEmptySlot);

    int numEmptySlots = 0; 
    deviceData->intBuffer.getDeviceObject(numEmptySlots, deviceData->numParticles);

    if (numEmptySlots == 0)
        return;

    thrust::device_ptr<int> firstFullSlot(deviceData->intBuffer2.buffer);
    thrust::exclusive_scan(firstFullSlot, firstFullSlot + deviceData->numParticles + 1, firstFullSlot);

    int numFullSlots = 0; 
    deviceData->intBuffer2.getDeviceObject(numFullSlots, deviceData->numParticles);

    deviceData->intBuffer3.resize(numFullSlots, false);

    kernel_listFullSlots << <(deviceData->numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (*deviceData, numFullSlots);

    if (numFullSlots > 0)
    {
        kernel_fillEmptySlots << <(deviceData->numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (*deviceData, numFullSlots);
    }

    deviceData->numParticles = numFullSlots;
}


void FluidDemo::addParticles() 
{
    // add source particles

    sourceParticlePos.clear();
    sourceParticleVel.clear();
    float sizeY = deviceData->sourceBounds.halfExtents.y;
    float sizeZ = deviceData->sourceBounds.halfExtents.z;

    float r = deviceData->particleRadius;
    int numY = (int)(sizeY / r);
    int numZ = (int)(sizeZ / r);
    Vec3 localVel = Vec3(deviceData->sourceSpeed, 0.0f, 0.0f);
    Vec3 worldVel = deviceData->sourceBounds.trans.rotate(localVel);

    for (int yi = -numY; yi <= numY; yi++) {
        for (int zi = -numZ; zi <= numZ; zi++) {
            Vec3 localPos = Vec3(0.0f, yi * r, zi * r);
            Vec3 worldPos = deviceData->sourceBounds.trans.transform(localPos);
            sourceParticlePos.push_back(worldPos);
        }
    }

    int numSourceParticles = Min((int)sourceParticlePos.size(), deviceData->maxParticles - deviceData->numParticles);

    if (numSourceParticles == 0)
        return;

    sourceParticleVel.resize(numSourceParticles, worldVel);

    deviceData->sourceParticlePos.set(sourceParticlePos);
    deviceData->sourceParticleVel.set(sourceParticleVel);

    Vec3 color(1.0f, 1.0f, 1.0f);
    float lifetime = 1.0f;

    kernel_addSourceParticles<<<(numSourceParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(*deviceData, numSourceParticles, color, lifetime);
    cudaCheck(cudaGetLastError());
}


void FluidDemo::updateCudaPhysics(float dt, cudaGraphicsResource* vboResource) {
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, &vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, vboResource);

    deviceData->vboData = d_vboData;

    // Handle source and sink (create new particles, remove particles in sink)
    removeParticles();
    addParticles();

    if (deviceData->numParticles == 0) {
        cudaGraphicsUnmapResources(1, &vboResource, 0);
        return;
    }

    int numBlocks = (deviceData->numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Recompute spatial hash
    hash->fillHash(deviceData->numParticles, deviceData->vboData, VBO_STRIDE, deviceData->gridSpacing);

    int numSubSteps = 5;
    float sdt = dt / (float)numSubSteps;

    auto meshesData = meshes->getDeviceData();
    auto hashData = hash->getDeviceData();

    for (int subStep = 0; subStep < numSubSteps; subStep++) {
        // Integrate
        kernel_integrate<<<numBlocks, THREADS_PER_BLOCK>>>(
            *deviceData, sdt, Vec3(0.0f, -demoDesc.gravity, 0.0f), demoDesc.maxVelocity);

        // Compute density using SPH
        kernel_computeDensity<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData, *hashData);

        deviceData->posCorr.setZero();

        // Particle-particle collision
        kernel_particleCollision_hash<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData, *hashData);
        cudaCheck(cudaGetLastError());

        const float relaxation = 0.3f;
        kernel_applyCorrections<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData, relaxation);

        // Wall collision
        kernel_wallCollision<<<numBlocks, THREADS_PER_BLOCK>>>(
            *deviceData, demoDesc.sceneBounds, deviceData->particleRadius);

        // Mesh collision
        if (meshesData->numMeshTriangles > 0) {
            float meshSearchRadius = 2.0f * deviceData->particleRadius;
            kernel_particleMeshesCollision<<<numBlocks, THREADS_PER_BLOCK>>>(
                *deviceData, *meshesData, meshSearchRadius, deviceData->particleRadius);
        }

        // Derive velocity from position change
        kernel_deriveVelocity<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData, sdt);
    }

    // Update marching cubes surface if in that render mode
    if (renderMode == FluidRenderMode::MarchingCubes && marchingCubesSurface) {
        marchingCubesSurface->update(deviceData->numParticles, deviceData->vboData, VBO_STRIDE, true);
    }

    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &vboResource, 0);
    cudaCheck(cudaGetLastError());
}


void FluidDemo::cleanupCudaPhysics(cudaGraphicsResource* vboResource) {
    if (!deviceData || deviceData->numParticles == 0) return;

    if (vboResource) {
        cudaGraphicsUnmapResources(1, &vboResource, 0);
    }

    if (deviceData) {
        deviceData->vboData = nullptr;
        deviceData->free();
        deviceData.reset();
    }

    if (meshes) {
        meshes->cleanup();
    }

    if (hash) {
        hash->cleanup();
    }

    if (vboResource) {
        cudaGraphicsUnregisterResource(vboResource);
    }

    cudaDeviceSynchronize();
}
