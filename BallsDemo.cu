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
        sortedPos.free();
        sortedPrevPos.free();
        ballBoundsLowers.free();
        ballBoundsUppers.free();
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
        s += sortedPos.allocationSize();
        s += sortedPrevPos.allocationSize();
        s += ballBoundsLowers.allocationSize();
        s += ballBoundsUppers.allocationSize();
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
    DeviceBuffer<Vec3> sortedPos;     // Positions in sorted order
    DeviceBuffer<Vec3> sortedPrevPos; // Previous positions in sorted order
    
    // BVH collision detection
    DeviceBuffer<Vec4> ballBoundsLowers;  // Lower bounds for BVH
    DeviceBuffer<Vec4> ballBoundsUppers;  // Upper bounds for BVH
    BVH bvh;                              // BVH structure
    
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
    data.vel.buffer[idx] += gravity * dt;
    
    // Apply friction
    data.vel.buffer[idx] *= friction;
    data.angVel.buffer[idx] *= friction;
    
    // Save previous position
    data.prevPos.buffer[idx] = pos;
    
    // Update position
    pos += data.vel.buffer[idx] * dt;
    
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
    data.ballBoundsLowers.buffer[idx] = Vec4(lower.x, lower.y, lower.z, 0.0f);
    data.ballBoundsUppers.buffer[idx] = Vec4(upper.x, upper.y, upper.z, 0.0f);
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
    
    data.hashVals.buffer[idx] = h;
    data.hashIds.buffer[idx] = idx;
}

// Kernel 3: Setup hash grid
__global__ void kernel_setupHash(BallsDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    unsigned int h = data.hashVals.buffer[idx];
    
    // Find cell boundaries
    if (idx == 0) {
        data.hashCellFirst.buffer[h] = 0;
    } else {
        unsigned int prevH = data.hashVals.buffer[idx - 1];
        if (h != prevH) {
            data.hashCellFirst.buffer[h] = idx;
            data.hashCellLast.buffer[prevH] = idx;
        }
    }
    
    if (idx == data.numBalls - 1) {
        data.hashCellLast.buffer[h] = data.numBalls;
    }
    
    // Copy positions to sorted arrays
    int sortedId = data.hashIds.buffer[idx];
    float* ballData = data.vboData + sortedId * 14;
    data.sortedPos.buffer[idx] = Vec3(ballData[0], ballData[1], ballData[2]);
    data.sortedPrevPos.buffer[idx] = data.prevPos.buffer[sortedId];
}

// Kernel 4: Ball-to-ball collision
__global__ void kernel_ballCollision(BallsDeviceData data, float bounce) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Get ball data
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
                
                int first = data.hashCellFirst.buffer[h];
                int last = data.hashCellLast.buffer[h];
                
                // Check all balls in this cell
                for (int i = first; i < last; i++) {
                    int otherIdx = data.hashIds.buffer[i];
                    if (otherIdx == idx) continue;
                    
                    // Get other ball data
                    float* otherData = data.vboData + otherIdx * 14;
                    Vec3 otherPos(otherData[0], otherData[1], otherData[2]);
                    float otherRadius = otherData[3];
                    
                    // Check collision
                    Vec3 delta = otherPos - pos;
                    float dist = delta.magnitude();
                    float minDist = radius + otherRadius;
                    
                    if (dist < minDist && dist > 0.001f) {
                        // Collision detected
                        Vec3 normal = delta / dist;
                        
                        // Separate balls (position correction)
                        float overlap = minDist - dist;
                        Vec3 correction = normal * (overlap * 0.5f);
                        
                        // Apply correction to position
                        pos -= correction;
                        
                        // Calculate relative velocity
                        Vec3 relVel = data.vel.buffer[otherIdx] - data.vel.buffer[idx];
                        float dvn = relVel.dot(normal);
                        
                        // Only resolve if balls are moving towards each other
                        if (dvn < 0) {
                            // Apply impulse
                            float impulse = dvn * bounce;
                            Vec3 impulseVec = normal * impulse;
                            data.vel.buffer[idx] += impulseVec;
                            
                            // Add spin from collision
                            float spinFactor = 0.3f;
                            Vec3 tangent = relVel - normal * dvn;
                            data.angVel.buffer[idx] += tangent.cross(normal) * spinFactor;
                        }
                    }
                }
            }
        }
    }
    
    // Write corrected position back
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
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
        data.vel.buffer[idx].x = -data.vel.buffer[idx].x * bounce;
        data.angVel.buffer[idx] += Vec3(0, data.vel.buffer[idx].z, -data.vel.buffer[idx].y) * 0.5f;
    }
    if (pos.x + radius > halfRoom) {
        pos.x = halfRoom - radius;
        data.vel.buffer[idx].x = -data.vel.buffer[idx].x * bounce;
        data.angVel.buffer[idx] += Vec3(0, -data.vel.buffer[idx].z, data.vel.buffer[idx].y) * 0.5f;
    }
    
    // Y axis (floor and ceiling)
    if (pos.y - radius < 0) {
        pos.y = radius;
        data.vel.buffer[idx].y = -data.vel.buffer[idx].y * bounce;
        data.angVel.buffer[idx] += Vec3(data.vel.buffer[idx].z, 0, -data.vel.buffer[idx].x) * 0.5f;
    }
    if (pos.y + radius > roomSize) {
        pos.y = roomSize - radius;
        data.vel.buffer[idx].y = -data.vel.buffer[idx].y * bounce;
        data.angVel.buffer[idx] += Vec3(-data.vel.buffer[idx].z, 0, data.vel.buffer[idx].x) * 0.5f;
    }
    
    // Z axis walls
    if (pos.z - radius < -halfRoom) {
        pos.z = -halfRoom + radius;
        data.vel.buffer[idx].z = -data.vel.buffer[idx].z * bounce;
        data.angVel.buffer[idx] += Vec3(-data.vel.buffer[idx].y, data.vel.buffer[idx].x, 0) * 0.5f;
    }
    if (pos.z + radius > halfRoom) {
        pos.z = halfRoom - radius;
        data.vel.buffer[idx].z = -data.vel.buffer[idx].z * bounce;
        data.angVel.buffer[idx] += Vec3(data.vel.buffer[idx].y, -data.vel.buffer[idx].x, 0) * 0.5f;
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
    
    // Get ball data
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
                if (otherIdx == idx) continue;
                
                // Get other ball data
                float* otherData = data.vboData + otherIdx * 14;
                Vec3 otherPos(otherData[0], otherData[1], otherData[2]);
                float otherRadius = otherData[3];
                
                // Check collision
                Vec3 delta = otherPos - pos;
                float dist = delta.magnitude();
                float minDist = radius + otherRadius;
                
                if (dist < minDist && dist > 0.001f) {
                    // Collision detected
                    Vec3 normal = delta / dist;
                    
                    // Separate balls (position correction)
                    float overlap = minDist - dist;
                    Vec3 correction = normal * (overlap * 0.5f);
                    
                    // Apply correction to position
                    pos -= correction;
                    
                    // Calculate relative velocity
                    Vec3 relVel = data.vel.buffer[otherIdx] - data.vel.buffer[idx];
                    float dvn = relVel.dot(normal);
                    
                    // Only resolve if balls are moving towards each other
                    if (dvn < 0) {
                        // Apply impulse
                        float impulse = dvn * bounce;
                        Vec3 impulseVec = normal * impulse;
                        data.vel.buffer[idx] += impulseVec;
                        
                        // Add spin from collision
                        float spinFactor = 0.3f;
                        Vec3 tangent = relVel - normal * dvn;
                        data.angVel.buffer[idx] += tangent.cross(normal) * spinFactor;
                    }
                }
            } else {  // Internal node
                // Push children onto stack
                if (stackCount < 63) {  // Prevent stack overflow
                    stack[stackCount++] = leftIndex;
                    stack[stackCount++] = rightIndex;
                }
            }
        }
    }
    
    // Write corrected position back
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}

// Kernel 6: Integrate quaternions
__global__ void kernel_integrateQuaternions(BallsDeviceData data, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * 14;
    
    // Read quaternion
    Quat quat(ballData[8], ballData[9], ballData[10], ballData[7]); // x, y, z, w
    
    // Integrate rotation
    Vec3 omega = data.angVel.buffer[idx] * dt;
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
    data.vel.buffer[idx].x = -2.0f + FRAND() * 4.0f;
    data.vel.buffer[idx].y = -2.0f + FRAND() * 4.0f;
    data.vel.buffer[idx].z = -2.0f + FRAND() * 4.0f;
    
    // Random angular velocity
    data.angVel.buffer[idx].x = -3.0f + FRAND() * 6.0f;
    data.angVel.buffer[idx].y = -3.0f + FRAND() * 6.0f;
    data.angVel.buffer[idx].z = -3.0f + FRAND() * 6.0f;
    
    // Store radius
    data.radii.buffer[idx] = ballData[3];
    
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
    g_deviceData.sortedPos.resize(numBalls, false);
    g_deviceData.sortedPrevPos.resize(numBalls, false);
    
    // Allocate BVH bounds buffers
    g_deviceData.ballBoundsLowers.resize(numBalls, false);
    g_deviceData.ballBoundsUppers.resize(numBalls, false);
    
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
        
        // BVH-based collision
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
        
        kernel_ballCollision<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, bounce);
    }
    
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

