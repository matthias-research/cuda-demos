#include "FluidDemo.h"
#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <vector>
#include "BVH.h"
#include "Geometry.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>



static const int HASH_SIZE = 37000111;  // Prime number for better distribution
static const int VBO_STRIDE = 6;    // pos(3), color(3)

// Device data structure - all simulation state on GPU
struct FluidDeviceData {
    void free()  // no destructor because cuda would call it in the kernels
    {
        numParticles = 0;
        numMeshTriangles = 0;

        vel.free();
        prevPos.free();
        hashVals.free();
        hashIds.free();
        hashCellFirst.free();
        hashCellLast.free();
        posCorr.free();

        meshFirstTriangle.free();
        meshVertices.free();
        meshTriIds.free();
        meshBoundsLower.free();
        meshBoundsUpper.free();
        meshTriBoundsLower.free();
        meshTriBoundsUpper.free();
        meshesBvh.free();
        trianglesBvh.free();
        rayCastMinT.free();

        meshesBvh.free();
        trianglesBvh.free();
    }
        
    int numParticles = 0;
    float particleRadius = 0.2f;
    float kernelRadius = 0.4f;
    float gridSpacing = 0.4f;
    int numMeshTriangles = 0;
    float worldOrig = -100.0f;
    
    // Physics data
    DeviceBuffer<Vec3> vel;           // Velocities
    DeviceBuffer<Vec3> prevPos;       // Previous positions (for PBD)
    
    // Hash grid
    DeviceBuffer<int> hashVals;       // Hash value per particle
    DeviceBuffer<int> hashIds;        // Particle index (gets sorted)
    DeviceBuffer<int> hashCellFirst;  // First particle in each cell
    DeviceBuffer<int> hashCellLast;   // Last particle in each cell
    
    // Mesh collision data
    DeviceBuffer<int> meshFirstTriangle; // Starting triangle index per mesh
    DeviceBuffer<Vec3> meshVertices;      // Concatenated vertices from all scene meshes
    DeviceBuffer<int> meshTriIds;         // Triangle indices (groups of 3)
    
    // Triangle BVH
    DeviceBuffer<Vec4> meshTriBoundsLower; // Lower bounds for each triangle
    DeviceBuffer<Vec4> meshTriBoundsUpper; // Upper bounds for each triangle
    DeviceBuffer<Vec4> meshBoundsLower;   // Overall mesh bounds
    DeviceBuffer<Vec4> meshBoundsUpper;   // Overall mesh bounds
    BVH meshesBvh;                        // BVH structure for entire meshes
    BVH trianglesBvh;                      // Separate BVH for triangles
    
    // Collision correction buffers (for symmetric resolution)
    DeviceBuffer<Vec3> posCorr;       // Position corrections

    DeviceBuffer<float> rayCastMinT; // Minimum t values for ray casting
    // VBO for positions (for rendering)
    float* vboData = nullptr;      // Interleaved: pos(3), color(3)    
};

#include "FluidDemo.h"

// No longer need factory functions; managed by FluidDemo class

// Helper device functions
__device__ inline int hashPosition(const Vec3& pos, float gridSpacing, float worldOrig) {
    int xi = floorf((pos.x - worldOrig) / gridSpacing);
    int yi = floorf((pos.y - worldOrig) / gridSpacing);
    int zi = floorf((pos.z - worldOrig) / gridSpacing);
    
    unsigned int h = abs((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) % HASH_SIZE;
    return h;
}

// Integrate - save previous position and predict new position (PBD)
__global__ void kernel_integrate(FluidDeviceData data, float dt, Vec3 gravity) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;
    
    // Read from VBO
    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);
    
    data.prevPos[idx] = pos;
    
    Vec3 vel = data.vel[idx];
    vel += gravity * dt;
    data.vel[idx] = vel;

    pos += vel * dt;
    
    particleData[0] = pos.x;
    particleData[1] = pos.y;
    particleData[2] = pos.z;
}

// Compute bounds for each triangle for mesh BVH construction
__global__ void kernel_computeTriangleBounds(FluidDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numMeshTriangles) return;
    
    // Get the three vertex indices for this triangle
    int i0 = data.meshTriIds[idx * 3 + 0];
    int i1 = data.meshTriIds[idx * 3 + 1];
    int i2 = data.meshTriIds[idx * 3 + 2];
    
    // Get the three vertices
    Vec3 v0 = data.meshVertices[i0];
    Vec3 v1 = data.meshVertices[i1];
    Vec3 v2 = data.meshVertices[i2];
    
    // Compute bounds
    Vec3 lower(
        fminf(fminf(v0.x, v1.x), v2.x),
        fminf(fminf(v0.y, v1.y), v2.y),
        fminf(fminf(v0.z, v1.z), v2.z)
    );
    
    Vec3 upper(
        fmaxf(fmaxf(v0.x, v1.x), v2.x),
        fmaxf(fmaxf(v0.y, v1.y), v2.y),
        fmaxf(fmaxf(v0.z, v1.z), v2.z)
    );
    
    // Store as Vec4 (BVH builder expects Vec4)
    data.meshTriBoundsLower[idx] = Vec4(lower.x, lower.y, lower.z, 0.0f);
    data.meshTriBoundsUpper[idx] = Vec4(upper.x, upper.y, upper.z, 0.0f);
}

// Fill hash values
__global__ void kernel_fillHash(FluidDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;
    
    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);
    
    int h = hashPosition(pos, data.gridSpacing, data.worldOrig);
    
    data.hashVals[idx] = h;
    data.hashIds[idx] = idx;
}

// Setup hash grid cell boundaries
__global__ void kernel_setupHash(FluidDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;
    
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
    
    if (idx == data.numParticles - 1) {
        data.hashCellLast[h] = data.numParticles;
    }
}

__device__ inline void separateHardParticles(
    int idx,
    int otherIdx,
    const Vec3& pos,
    const Vec3& otherPos,
    FluidDeviceData& data)
{
    if (idx >= otherIdx) return;
    
    Vec3 delta = otherPos - pos;
    float dist = delta.magnitude();
    float minDist = data.particleRadius * 2.0f;
    
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

__global__ void kernel_separateHardParticles(FluidDeviceData data, float bounce) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;
    
    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    float invGridSpacing = 1.0f / data.gridSpacing;
    
    // Compute grid cell
    int xi = floorf((pos.x - data.worldOrig) * invGridSpacing);
    int yi = floorf((pos.y - data.worldOrig) * invGridSpacing);
    int zi = floorf((pos.z - data.worldOrig) * invGridSpacing);
    
    // Check neighboring cells (3x3x3 = 27 cells)
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int cellX = xi + dx;
                int cellY = yi + dy;
                int cellZ = zi + dz;
                
                unsigned int h = abs((cellX * 92837111) ^ (cellY * 689287499) ^ (cellZ * 283923481)) % HASH_SIZE;
                
                int first = data.hashCellFirst[h];
                int last = data.hashCellLast[h];
                
                // Check all particles in this cell
                for (int i = first; i < last; i++) {
                    int otherIdx = data.hashIds[i];
                    
                    // Get other particle data (read from VBO using index)
                    float* otherData = data.vboData + otherIdx * VBO_STRIDE;
                    Vec3 otherPos(otherData[0], otherData[1], otherData[2]);
                    
                    // Handle collision using symmetric resolution
                    separateHardParticles(idx, otherIdx, pos, otherPos, data);
                }
            }
        }
    }
}

__global__ void kernel_wallCollision(FluidDeviceData data, Bounds3 sceneBounds) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) 
        return;
        
    Plane walls[5] = {
        Plane(Vec3(-1, 0, 0),  -sceneBounds.maximum.x),
        Plane(Vec3(1, 0, 0),   sceneBounds.minimum.x),
        Plane(Vec3(0, 1, 0),   sceneBounds.minimum.y),
        Plane(Vec3(0, 0, -1),  -sceneBounds.maximum.z),
        Plane(Vec3(0, 0, 1),   sceneBounds.minimum.z)
    };
    
    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);

    float radius = data.particleRadius;
    
    for (int i = 0; i < 5; i++) {
        const Plane& wall = walls[i];

        float signedDist = wall.n.dot(pos) - radius - wall.d;
    
        if (signedDist < 0) {
            pos -= wall.n * signedDist;
        }
    }   
    
    particleData[0] = pos.x;
    particleData[1] = pos.y;
    particleData[2] = pos.z;
}

__device__ void kernel_particleMeshCollision(FluidDeviceData data, int particleIdx, int meshIdx, float searchRadius) {

    float* particleData = data.vboData + particleIdx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);
    float radius = data.particleRadius;
    
    Bounds3 queryBounds(pos - Vec3(searchRadius, searchRadius, searchRadius), 
                       pos + Vec3(searchRadius, searchRadius, searchRadius));
    
    int rootNode = data.trianglesBvh.mRootNodes[meshIdx];
    if (rootNode < 0) 
        return;
        
    // Stack-based BVH traversal
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        // Get node bounds
        PackedNodeHalf lower = data.trianglesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = data.trianglesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z),
            Vec3(upper.x, upper.y, upper.z));

        // Test intersection with query bounds
        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf node - contains a triangle
                int triIdx = leftIndex;

                // Get the three vertex indices for this triangle
                int i0 = data.meshTriIds[triIdx * 3 + 0];
                int i1 = data.meshTriIds[triIdx * 3 + 1];
                int i2 = data.meshTriIds[triIdx * 3 + 2];

                // Get the three vertices
                Vec3 v0 = data.meshVertices[i0];
                Vec3 v1 = data.meshVertices[i1];
                Vec3 v2 = data.meshVertices[i2];

                // Compute closest point on triangle to particle position
                Vec3 baryCoords = getClosestPointOnTriangle(pos, v0, v1, v2);
                Vec3 closestPoint = v0 * baryCoords.x + v1 * baryCoords.y + v2 * baryCoords.z;

                Vec3 n = (pos - closestPoint);
                float dist = n.normalize();

                if (dist < radius)
                {
                    pos += n * (radius - dist);
                }
            }
            else {  // Internal node
                if (stackCount < 63) {  // Prevent stack overflow
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

// Particle-meshes collision using BVH
__global__ void kernel_particleMeshesCollision(FluidDeviceData data, float searchRadius)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;
    
    if (data.numMeshTriangles == 0) return;
    
    float* particleData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(particleData[0], particleData[1], particleData[2]);
    
    // Query bounds (particle position + search radius)
    Bounds3 queryBounds(pos - Vec3(searchRadius, searchRadius, searchRadius), 
                       pos + Vec3(searchRadius, searchRadius, searchRadius));
    
    // Get root node of meshes BVH
    int rootNode = data.meshesBvh.mRootNodes[0];
    if (rootNode < 0) return;
        
    // Stack-based BVH traversal
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        // Get node bounds
        PackedNodeHalf lower = data.meshesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = data.meshesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z),
            Vec3(upper.x, upper.y, upper.z));

        // Test intersection with query bounds
        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf node - contains a mesh
                int meshIdx = leftIndex;
                kernel_particleMeshCollision(data, idx, meshIdx, searchRadius);
            }

            else {  // Internal node
             // Push children onto stack
                if (stackCount < 63) {  // Prevent stack overflow
                    stack[stackCount++] = leftIndex;
                    stack[stackCount++] = rightIndex;
                }
            }
        }
    }
}

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


__global__ void kernel_initParticles(FluidDeviceData data, Bounds3 particleBounds,
                                  int particlesPerLayer, int particlesPerRow, int particlesPerCol, float gridSpacing, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numParticles) return;
    
    float* particleData = data.vboData + idx * VBO_STRIDE;
    
    int layer = idx / particlesPerLayer;
    int inLayer = idx % particlesPerLayer;
    int row = inLayer / particlesPerRow;
    int col = inLayer % particlesPerRow;
    
    particleData[0] = particleBounds.minimum.x + gridSpacing + col * gridSpacing;  // x
    particleData[2] = particleBounds.minimum.z + gridSpacing + row * gridSpacing;  // z
    particleData[1] = particleBounds.minimum.y + gridSpacing + layer * gridSpacing; // y
}



// Particle - mesh collision using BVH
__device__ bool kernel_raycastMesh(FluidDeviceData data, int meshIdx, Ray ray, float& minT) {

    int rootNode = data.trianglesBvh.mRootNodes[meshIdx];
    if (rootNode < 0)
        return false;
        
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    bool hit = false;

    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        // Get node bounds
        PackedNodeHalf lower = data.trianglesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = data.trianglesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z),
            Vec3(upper.x, upper.y, upper.z));
        
        if (rayBoundsIntersection(nodeBounds, ray))
        {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf node - contains a triangle
                int triIdx = leftIndex;

                // Get the three vertex indices for this triangle
                int i0 = data.meshTriIds[triIdx * 3 + 0];
                int i1 = data.meshTriIds[triIdx * 3 + 1];
                int i2 = data.meshTriIds[triIdx * 3 + 2];

                // Get the three vertices
                Vec3 v0 = data.meshVertices[i0];
                Vec3 v1 = data.meshVertices[i1];
                Vec3 v2 = data.meshVertices[i2];

                float t, u, v;

                if (rayTriangleIntersection(ray, v0, v1, v2, t, u, v))
                {
                    hit = true;
                    if (t > 0.0f && t < minT)
                    {
                        minT = t;
                    }
                }
            }
            else {  // Internal node
             // Push children onto stack
                if (stackCount < 63) {  // Prevent stack overflow
                    stack[stackCount++] = leftIndex;
                    stack[stackCount++] = rightIndex; 
                }
            }
        }
    }
    return hit;
}


__global__ void kernel_raycastMeshes(FluidDeviceData data, Ray ray)
{
    float minT = MaxFloat;
    int stack[64];
    stack[0] = data.meshesBvh.mRootNodes[0];
    int count = 1;

    while (count)
    {
        const int nodeIndex = stack[--count];

        PackedNodeHalf lower = data.meshesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = data.meshesBvh.mNodeUppers[nodeIndex];

        Bounds3 bounds(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

        if (rayBoundsIntersection(bounds, ray))
        {
           const int leftIndex = lower.i;
           const int rightIndex = upper.i;

           if (lower.b)
           {
               kernel_raycastMesh(data, leftIndex, ray, minT);
           }
           else
           {
               stack[count++] = leftIndex;
               stack[count++] = rightIndex;
           }
        }
    }
    data.rayCastMinT[0] = minT;
}

// Host functions callable from FluidDemo.cpp


void FluidDemo::initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, Scene* scene) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before init: %s\n", cudaGetErrorString(err));
    }

    if (!deviceData) 
        deviceData = std::make_shared<FluidDeviceData>();
    if (!bvhBuilder) 
        bvhBuilder = std::make_shared<BVHBuilder>();

    deviceData->numParticles = demoDesc.numParticles;

    // Allocate physics arrays using DeviceBuffer
    deviceData->vel.resize(demoDesc.numParticles, false);
    deviceData->prevPos.resize(demoDesc.numParticles, false);

    // Allocate hash grid using DeviceBuffer
    deviceData->hashVals.resize(demoDesc.numParticles, false);
    deviceData->hashIds.resize(demoDesc.numParticles, false);
    deviceData->hashCellFirst.resize(HASH_SIZE, false);
    deviceData->hashCellLast.resize(HASH_SIZE, false);

    // Allocate collision correction buffers
    deviceData->posCorr.resize(demoDesc.numParticles, false);

    // Initialize memory
    deviceData->vel.setZero();
    deviceData->hashCellFirst.setZero();
    deviceData->hashCellLast.setZero();

    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // Map VBO and initialize particles on GPU
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, *vboResource);
    deviceData->vboData = d_vboData;

    // Launch kernel to initialize positions/colors (not shown here)
    // kernel_initParticles<<<...>>>(*deviceData, ...);

    cudaGraphicsUnmapResources(1, vboResource, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA initialization error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void FluidDemo::updateCudaPhysics(float dt, cudaGraphicsResource* vboResource) {
    if (deviceData->numParticles == 0) return;
    int numBlocks = (deviceData->numParticles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Map VBO
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, &vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, vboResource);
    deviceData->vboData = d_vboData;

    // Hash grid collision detection
    kernel_fillHash<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData);
    thrust::device_ptr<int> hashVals(deviceData->hashVals.buffer);
    thrust::device_ptr<int> hashIds(deviceData->hashIds.buffer);
    thrust::sort_by_key(hashVals, hashVals + deviceData->numParticles, hashIds);
    deviceData->hashCellFirst.setZero();
    deviceData->hashCellLast.setZero();
    kernel_setupHash<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData);

    // Physics pipeline
    kernel_integrate<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData, dt, Vec3(0.0f, -demoDesc.gravity, 0.0f));
    deviceData->posCorr.setZero();
    kernel_separateHardParticles <<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData, demoDesc.particleRadius);
    // Optionally: mesh collision kernel

    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &vboResource, 0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA physics error: %s\n", cudaGetErrorString(err));
    }
}

void FluidDemo::cleanupCudaPhysics(cudaGraphicsResource* vboResource) {
    if (!deviceData || deviceData->numParticles == 0) return;
    cudaDeviceSynchronize();
    if (vboResource) {
        cudaGraphicsUnmapResources(1, &vboResource, 0);
    }
    if (deviceData) {
        deviceData->free();
        deviceData.reset();
    }
    if (bvhBuilder) {
        bvhBuilder.reset();
    }
    if (vboResource) {
        cudaGraphicsUnregisterResource(vboResource);
    }
    cudaDeviceSynchronize();
}
