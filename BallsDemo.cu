#include "BallsDemo.h"
#include "CudaUtils.h"
#include "BVH.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <cstdio>

// Hash grid parameters
static const int HASH_SIZE = 370111;  // Prime number for better distribution

// Device data structure - all simulation state on GPU
struct BallsDeviceData {
    void free()  // no destructor because cuda would call it in the kernels
    {
        vel.free();
        angVel.free();
        prevPos.free();
        radii.free();
        hashVals.free();
        hashIds.free();
        hashCellFirst.free();
        hashCellLast.free();
        ballBoundsLowers.free();
        ballBoundsUppers.free();
        posCorr.free();
        velCorr.free();
        angVelCorr.free();
        numBalls = 0;
    }
    
    size_t allocationSize() const
    {
        size_t s = 0;
        s += vel.allocationSize();
        s += angVel.allocationSize();
        s += prevPos.allocationSize();
        s += radii.allocationSize();
        s += hashVals.allocationSize();
        s += hashIds.allocationSize();
        s += hashCellFirst.allocationSize();
        s += hashCellLast.allocationSize();
        s += ballBoundsLowers.allocationSize();
        s += ballBoundsUppers.allocationSize();
        s += posCorr.allocationSize();
        s += velCorr.allocationSize();
        s += angVelCorr.allocationSize();
        return s;
    }
    
    int numBalls = 0;
    float gridSpacing = 0.0f;
    float worldOrig = 0.0f;
    
    // Physics data
    DeviceBuffer<Vec3> vel;           // Velocities
    DeviceBuffer<Vec3> angVel;        // Angular velocities
    DeviceBuffer<Vec3> prevPos;       // Previous positions (for position-based dynamics)
    DeviceBuffer<float> radii;        // Radius per ball (for variable sizes)
    
    // Hash grid
    DeviceBuffer<int> hashVals;       // Hash value per ball
    DeviceBuffer<int> hashIds;        // Ball index (gets sorted)
    DeviceBuffer<int> hashCellFirst;  // First ball in each cell
    DeviceBuffer<int> hashCellLast;   // Last ball in each cell
    
    // BVH collision detection
    DeviceBuffer<Vec4> ballBoundsLowers;  // Lower bounds for BVH
    DeviceBuffer<Vec4> ballBoundsUppers;  // Upper bounds for BVH
    BVH bvh;                              // BVH structure
    
    // Collision correction buffers (for symmetric resolution)
    DeviceBuffer<Vec3> posCorr;       // Position corrections
    DeviceBuffer<Vec3> velCorr;       // Velocity corrections
    DeviceBuffer<Vec3> angVelCorr;    // Angular velocity corrections
    
    // VBO data (mapped from OpenGL)
    float* vboData = nullptr;      // Interleaved: pos(3), radius(1), color(3), quat(4), pad(3) = 14 floats
};

// Global device data (stored on host, contains device pointers)
static BallsDeviceData g_deviceData;

// Helper device functions
__device__ inline int hashPosition(const Vec3& pos, float gridSpacing, float worldOrig) {
    int xi = floorf((pos.x - worldOrig) / gridSpacing);
    int yi = floorf((pos.y - worldOrig) / gridSpacing);
    int zi = floorf((pos.z - worldOrig) / gridSpacing);
    
    unsigned int h = abs((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) % HASH_SIZE;
    return h;
}

// Kernel 1: Integrate - apply gravity and update positions
__global__ void kernel_integrate(BallsDeviceData data, float dt, Vec3 gravity, float friction) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read from VBO
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Apply gravity
    data.vel[idx] += gravity * dt;
    
    // Apply friction
    data.vel[idx] *= friction;
    data.angVel[idx] *= friction;
    
    // Save previous position
    data.prevPos[idx] = pos;
    
    // Update position
    pos += data.vel[idx] * dt;
    
    // Write position back to VBO
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}

// Kernel: Compute bounds for each ball for BVH construction
__global__ void kernel_computeBallBounds(BallsDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read from VBO
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    // Create bounds for this ball
    Vec3 lower = pos - Vec3(radius, radius, radius);
    Vec3 upper = pos + Vec3(radius, radius, radius);
    
    // Store as Vec4 (BVH builder expects Vec4)
    data.ballBoundsLowers[idx] = Vec4(lower.x, lower.y, lower.z, 0.0f);
    data.ballBoundsUppers[idx] = Vec4(upper.x, upper.y, upper.z, 0.0f);
}

// Kernel 2: Fill hash values
__global__ void kernel_fillHash(BallsDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read position from VBO
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Compute hash
    int h = hashPosition(pos, data.gridSpacing, data.worldOrig);
    
    data.hashVals[idx] = h;
    data.hashIds[idx] = idx;
}

// Kernel 3: Setup hash grid cell boundaries
__global__ void kernel_setupHash(BallsDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
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
    
    if (idx == data.numBalls - 1) {
        data.hashCellLast[h] = data.numBalls;
    }
}

// Device function: Handle collision between two balls (symmetric resolution)
// Only processes if idx < otherIdx to handle each pair once
__device__ inline void handleBallCollision(
    int idx,
    int otherIdx,
    const Vec3& pos,
    float radius,
    const Vec3& otherPos,
    float otherRadius,
    float bounce,
    BallsDeviceData& data)
{
    // Only process each pair once (symmetric resolution)
    if (idx >= otherIdx) return;
    
    // Check collision
    Vec3 delta = otherPos - pos;
    float dist = delta.magnitude();
    float minDist = radius + otherRadius;
    
    if (dist < minDist && dist > 0.001f) {
        // Collision detected
        Vec3 normal = delta / dist;
        
        // Position correction: separate balls symmetrically
        float overlap = minDist - dist;
        Vec3 separation = normal * (overlap * 0.5f);
        
        atomicAdd(&data.posCorr[idx].x, -separation.x);
        atomicAdd(&data.posCorr[idx].y, -separation.y);
        atomicAdd(&data.posCorr[idx].z, -separation.z);
        
        atomicAdd(&data.posCorr[otherIdx].x, separation.x);
        atomicAdd(&data.posCorr[otherIdx].y, separation.y);
        atomicAdd(&data.posCorr[otherIdx].z, separation.z);
        
        // Velocity correction: calculate relative velocity and impulse
        Vec3 relVel = data.vel[otherIdx] - data.vel[idx];
        float dvn = relVel.dot(normal);
        
        // Only resolve if balls are moving towards each other
        if (dvn < 0) {
            // Apply impulse (symmetric: equal and opposite)
            float impulse = dvn * bounce;
            Vec3 impulseVec = normal * impulse;
            
            atomicAdd(&data.velCorr[idx].x, impulseVec.x);
            atomicAdd(&data.velCorr[idx].y, impulseVec.y);
            atomicAdd(&data.velCorr[idx].z, impulseVec.z);
            
            atomicAdd(&data.velCorr[otherIdx].x, -impulseVec.x);
            atomicAdd(&data.velCorr[otherIdx].y, -impulseVec.y);
            atomicAdd(&data.velCorr[otherIdx].z, -impulseVec.z);
            
            // Angular velocity correction: add spin from collision
            float spinFactor = 0.3f;
            Vec3 tangent = relVel - normal * dvn;
            Vec3 spin = tangent.cross(normal) * spinFactor;
            
            atomicAdd(&data.angVelCorr[idx].x, spin.x);
            atomicAdd(&data.angVelCorr[idx].y, spin.y);
            atomicAdd(&data.angVelCorr[idx].z, spin.z);
            
            atomicAdd(&data.angVelCorr[otherIdx].x, -spin.x);
            atomicAdd(&data.angVelCorr[otherIdx].y, -spin.y);
            atomicAdd(&data.angVelCorr[otherIdx].z, -spin.z);
        }
    }
}

// Kernel 4: Ball-to-ball collision (hash grid)
__global__ void kernel_ballCollision(BallsDeviceData data, float bounce) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Get ball data (read from VBO)
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    // Compute grid cell
    int xi = floorf((pos.x - data.worldOrig) / data.gridSpacing);
    int yi = floorf((pos.y - data.worldOrig) / data.gridSpacing);
    int zi = floorf((pos.z - data.worldOrig) / data.gridSpacing);
    
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
                
                // Check all balls in this cell
                for (int i = first; i < last; i++) {
                    int otherIdx = data.hashIds[i];
                    
                    // Get other ball data (read from VBO using index)
                    float* otherData = data.vboData + otherIdx * 14;
                    Vec3 otherPos(otherData[0], otherData[1], otherData[2]);
                    float otherRadius = otherData[3];
                    
                    // Handle collision using symmetric resolution
                    handleBallCollision(idx, otherIdx, pos, radius, otherPos, otherRadius, bounce, data);
                }
            }
        }
    }
}

// Kernel 5: Wall collision
__global__ void kernel_wallCollision(BallsDeviceData data, float roomSize, float bounce) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    float halfRoom = roomSize * 0.5f;
    
    // X axis walls
    if (pos.x - radius < -halfRoom) {
        pos.x = -halfRoom + radius;
        data.vel[idx].x = -data.vel[idx].x * bounce;
        data.angVel[idx] += Vec3(0, data.vel[idx].z, -data.vel[idx].y) * 0.5f;
    }
    if (pos.x + radius > halfRoom) {
        pos.x = halfRoom - radius;
        data.vel[idx].x = -data.vel[idx].x * bounce;
        data.angVel[idx] += Vec3(0, -data.vel[idx].z, data.vel[idx].y) * 0.5f;
    }
    
    // Y axis (floor and ceiling)
    if (pos.y - radius < 0) {
        pos.y = radius;
        data.vel[idx].y = -data.vel[idx].y * bounce;
        data.angVel[idx] += Vec3(data.vel[idx].z, 0, -data.vel[idx].x) * 0.5f;
    }
    if (pos.y + radius > roomSize) {
        pos.y = roomSize - radius;
        data.vel[idx].y = -data.vel[idx].y * bounce;
        data.angVel[idx] += Vec3(-data.vel[idx].z, 0, data.vel[idx].x) * 0.5f;
    }
    
    // Z axis walls
    if (pos.z - radius < -halfRoom) {
        pos.z = -halfRoom + radius;
        data.vel[idx].z = -data.vel[idx].z * bounce;
        data.angVel[idx] += Vec3(-data.vel[idx].y, data.vel[idx].x, 0) * 0.5f;
    }
    if (pos.z + radius > halfRoom) {
        pos.z = halfRoom - radius;
        data.vel[idx].z = -data.vel[idx].z * bounce;
        data.angVel[idx] += Vec3(data.vel[idx].y, -data.vel[idx].x, 0) * 0.5f;
    }
    
    // Write back
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}

// Kernel: BVH-based ball collision (alternative to hash grid)
__global__ void kernel_ballCollision_BVH(BallsDeviceData data, float bounce) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Get ball data (read from VBO)
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    // Query bounds (ball position + radius)
    Bounds3 queryBounds(pos - Vec3(radius, radius, radius), 
                       pos + Vec3(radius, radius, radius));
    
    // Get root node
    int rootNode = data.bvh.mRootNodes ? data.bvh.mRootNodes[0] : -1;
    if (rootNode < 0) return;
    
    // Stack-based BVH traversal
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];
        
        // Get node bounds
        PackedNodeHalf lower = data.bvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = data.bvh.mNodeUppers[nodeIndex];
        
        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z), 
                          Vec3(upper.x, upper.y, upper.z));
        
        // Test intersection with query bounds
        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;
            
            if (lower.b) {  // Leaf node
                int otherIdx = leftIndex;
                
                // Get other ball data (read from VBO using index)
                float* otherData = data.vboData + otherIdx * 14;
                Vec3 otherPos(otherData[0], otherData[1], otherData[2]);
                float otherRadius = otherData[3];
                
                // Handle collision using symmetric resolution
                handleBallCollision(idx, otherIdx, pos, radius, otherPos, otherRadius, bounce, data);
            } else {  // Internal node
                // Push children onto stack
                if (stackCount < 63) {  // Prevent stack overflow
                    stack[stackCount++] = leftIndex;
                    stack[stackCount++] = rightIndex;
                }
            }
        }
    }
}

// Kernel: Apply collision corrections with relaxation
__global__ void kernel_applyCorrections(BallsDeviceData data, float relaxation) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read from VBO
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Apply corrections with relaxation factor
    pos += data.posCorr[idx] * relaxation;
    data.vel[idx] += data.velCorr[idx] * relaxation;
    data.angVel[idx] += data.angVelCorr[idx] * relaxation;
    
    // Write position back to VBO
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}

// Kernel: Zero correction buffers
__global__ void kernel_zeroCorrections(BallsDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    data.posCorr[idx] = Vec3(0, 0, 0);
    data.velCorr[idx] = Vec3(0, 0, 0);
    data.angVelCorr[idx] = Vec3(0, 0, 0);
}

// Kernel 6: Integrate quaternions
__global__ void kernel_integrateQuaternions(BallsDeviceData data, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * 14;
    
    // Read quaternion
    Quat quat(ballData[8], ballData[9], ballData[10], ballData[7]); // x, y, z, w
    
    // Integrate rotation
    Vec3 omega = data.angVel[idx] * dt;
    quat = quat.rotateLinear(quat, omega);
    
    // Write back
    ballData[7] = quat.w;
    ballData[8] = quat.x;
    ballData[9] = quat.y;
    ballData[10] = quat.z;
}

// Initialization kernel - sets up random balls on GPU
__global__ void kernel_initBalls(BallsDeviceData data, float roomSize, unsigned long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Fast pseudo-random using hash functions (much faster than curand_init)
    unsigned int hash = idx + seed;
    hash = (hash ^ 61) ^ (hash >> 16);
    hash = hash + (hash << 3);
    hash = hash ^ (hash >> 4);
    hash = hash * 0x27d4eb2d;
    hash = hash ^ (hash >> 15);
    
    // Simple LCG for random numbers
    #define FRAND() (hash = hash * 1664525u + 1013904223u, float(hash) / float(0xFFFFFFFFu))
    
    float* ballData = data.vboData + idx * 14;
    
    // Random position
    ballData[0] = -roomSize * 0.4f + FRAND() * roomSize * 0.8f;  // x
    ballData[1] = 1.0f + FRAND() * roomSize * 0.5f;              // y
    ballData[2] = -roomSize * 0.4f + FRAND() * roomSize * 0.8f;  // z
    
    // Random radius
    ballData[3] = (0.1f + FRAND() * 0.3f) * 1.0f;
    
    // Random color (vibrant)
    ballData[4] = 0.3f + FRAND() * 0.7f;  // r
    ballData[5] = 0.3f + FRAND() * 0.7f;  // g
    ballData[6] = 0.3f + FRAND() * 0.7f;  // b
    
    // Identity quaternion
    ballData[7] = 1.0f;  // w
    ballData[8] = 0.0f;  // x
    ballData[9] = 0.0f;  // y
    ballData[10] = 0.0f; // z
    
    // Padding
    ballData[11] = 0.0f;
    ballData[12] = 0.0f;
    ballData[13] = 0.0f;
    
    // Random velocity
    data.vel[idx].x = -2.0f + FRAND() * 4.0f;
    data.vel[idx].y = -2.0f + FRAND() * 4.0f;
    data.vel[idx].z = -2.0f + FRAND() * 4.0f;
    
    // Random angular velocity
    data.angVel[idx].x = -3.0f + FRAND() * 6.0f;
    data.angVel[idx].y = -3.0f + FRAND() * 6.0f;
    data.angVel[idx].z = -3.0f + FRAND() * 6.0f;
    
    // Store radius
    data.radii[idx] = ballData[3];
    
    #undef FRAND
}

// Host functions callable from BallsDemo.cpp

static BVHBuilder* g_bvhBuilder = nullptr;

extern "C" void initCudaPhysics(int numBalls, float roomSize, GLuint vbo, cudaGraphicsResource** vboResource, BVHBuilder* bvhBuilder) {
    g_deviceData.numBalls = numBalls;
    g_deviceData.worldOrig = -100.0f;
    g_bvhBuilder = bvhBuilder;
    
    // Find maximum radius (updated to match smaller balls)
    float maxRadius = (0.1f + 0.3f) * 1.0f;  // Maximum possible radius
    g_deviceData.gridSpacing = 2.5f * maxRadius;
    
    // Allocate physics arrays using DeviceBuffer
    g_deviceData.vel.resize(numBalls, false);
    g_deviceData.angVel.resize(numBalls, false);
    g_deviceData.prevPos.resize(numBalls, false);
    g_deviceData.radii.resize(numBalls, false);
    
    // Allocate hash grid using DeviceBuffer
    g_deviceData.hashVals.resize(numBalls, false);
    g_deviceData.hashIds.resize(numBalls, false);
    g_deviceData.hashCellFirst.resize(HASH_SIZE, false);
    g_deviceData.hashCellLast.resize(HASH_SIZE, false);
    
    // Allocate BVH bounds buffers
    g_deviceData.ballBoundsLowers.resize(numBalls, false);
    g_deviceData.ballBoundsUppers.resize(numBalls, false);
    
    // Allocate collision correction buffers
    g_deviceData.posCorr.resize(numBalls, false);
    g_deviceData.velCorr.resize(numBalls, false);
    g_deviceData.angVelCorr.resize(numBalls, false);
    
    // Initialize memory
    g_deviceData.vel.setZero();
    g_deviceData.angVel.setZero();
    g_deviceData.prevPos.setZero();
    g_deviceData.hashCellFirst.setZero();
    g_deviceData.hashCellLast.setZero();
    
    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    
    // Map VBO and initialize balls on GPU
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, *vboResource);
    
    // Update VBO pointer
    g_deviceData.vboData = d_vboData;
    
    // Initialize balls with random data
    int numBlocks = (numBalls + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching kernel_initBalls with %d blocks, %d threads per block, %d total balls\n", numBlocks, THREADS_PER_BLOCK, numBalls);
    
    kernel_initBalls<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, roomSize, time(nullptr));
    
    // Check for kernel launch errors
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(launchErr));
    }
    
    printf("Waiting for kernel to complete...\n");
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(syncErr));
    }
    printf("Kernel completed!\n");
    
    cudaGraphicsUnmapResources(1, vboResource, 0);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA initialization error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void updateCudaPhysics(float dt, Vec3 gravity, float friction, float bounce, float roomSize, 
                                   cudaGraphicsResource* vboResource, bool useBVH) {
    if (g_deviceData.numBalls == 0) return;
    
    int numBlocks = (g_deviceData.numBalls + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Map VBO
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, &vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, vboResource);
    
    // Update VBO pointer
    g_deviceData.vboData = d_vboData;
    
    // Run physics pipeline
    kernel_integrate<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, dt, gravity, friction);
    
    // Zero correction buffers before collision detection
    kernel_zeroCorrections<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData);
    
    if (useBVH) {
        // BVH-based collision detection
        
        // Compute bounds for each ball
        kernel_computeBallBounds<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData);
        
        // Build BVH
        if (g_bvhBuilder) {
            g_bvhBuilder->build(g_deviceData.bvh,
                              g_deviceData.ballBoundsLowers.buffer,
                              g_deviceData.ballBoundsUppers.buffer,
                              g_deviceData.numBalls,
                              nullptr,  // No grouping
                              0);
        }
        
        // BVH-based collision (accumulates corrections)
        kernel_ballCollision_BVH<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, bounce);
    } else {
        // Hash grid collision detection
        
        kernel_fillHash<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData);
        
        // Sort by hash
        thrust::device_ptr<int> hashVals(g_deviceData.hashVals.buffer);
        thrust::device_ptr<int> hashIds(g_deviceData.hashIds.buffer);
        thrust::sort_by_key(hashVals, hashVals + g_deviceData.numBalls, hashIds);
        
        g_deviceData.hashCellFirst.setZero();
        g_deviceData.hashCellLast.setZero();
        
        kernel_setupHash<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData);
        
        // Hash grid collision (accumulates corrections)
        kernel_ballCollision<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, bounce);
    }
    
    // Apply corrections with relaxation
    const float relaxation = 0.3f;
    kernel_applyCorrections<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, relaxation);
    
    kernel_wallCollision<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, roomSize, bounce);
    
    kernel_integrateQuaternions<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, dt);
    
    // Unmap VBO
    cudaGraphicsUnmapResources(1, &vboResource, 0);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA physics error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void cleanupCudaPhysics(cudaGraphicsResource* vboResource) {
    if (g_deviceData.numBalls == 0) return;
    
    // Free all device memory using DeviceBuffer::free()
    g_deviceData.free();
    
    // Free BVH
    g_deviceData.bvh.free();
    
    // Zero out VBO pointer
    g_deviceData.vboData = nullptr;
    g_bvhBuilder = nullptr;
    
    // Unregister VBO
    if (vboResource) {
        cudaGraphicsUnregisterResource(vboResource);
    }
}

