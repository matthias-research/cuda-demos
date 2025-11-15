#include "BallsDemo.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <curand_kernel.h>
#include <cstdio>

// Hash grid parameters
static const int HASH_SIZE = 370111;  // Prime number for better distribution
static const int THREADS_PER_BLOCK = 256;

// Device data structure - all simulation state on GPU
struct BallsDeviceData {
    int numBalls;
    float gridSpacing;
    float worldOrig;
    
    // Physics data
    Vec3* vel;           // Velocities
    Vec3* angVel;        // Angular velocities
    Vec3* prevPos;       // Previous positions (for position-based dynamics)
    float* radii;        // Radius per ball (for variable sizes)
    
    // Hash grid
    int* hashVals;       // Hash value per ball
    int* hashIds;        // Ball index (gets sorted)
    int* hashCellFirst;  // First ball in each cell
    int* hashCellLast;   // Last ball in each cell
    Vec3* sortedPos;     // Positions in sorted order
    Vec3* sortedPrevPos; // Previous positions in sorted order
    
    // VBO data (mapped from OpenGL)
    float* vboData;      // Interleaved: pos(3), radius(1), color(3), quat(4), pad(3) = 14 floats
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

// Kernel 3: Setup hash grid
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
    
    // Copy positions to sorted arrays
    int sortedId = data.hashIds[idx];
    float* ballData = data.vboData + sortedId * 14;
    data.sortedPos[idx] = Vec3(ballData[0], ballData[1], ballData[2]);
    data.sortedPrevPos[idx] = data.prevPos[sortedId];
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
                
                int first = data.hashCellFirst[h];
                int last = data.hashCellLast[h];
                
                // Check all balls in this cell
                for (int i = first; i < last; i++) {
                    int otherIdx = data.hashIds[i];
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
                        Vec3 relVel = data.vel[otherIdx] - data.vel[idx];
                        float dvn = relVel.dot(normal);
                        
                        // Only resolve if balls are moving towards each other
                        if (dvn < 0) {
                            // Apply impulse
                            float impulse = dvn * bounce;
                            Vec3 impulseVec = normal * impulse;
                            data.vel[idx] += impulseVec;
                            
                            // Add spin from collision
                            float spinFactor = 0.3f;
                            Vec3 tangent = relVel - normal * dvn;
                            data.angVel[idx] += tangent.cross(normal) * spinFactor;
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
    
    // Initialize random state
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    float* ballData = data.vboData + idx * 14;
    
    // Random position
    ballData[0] = -roomSize * 0.4f + curand_uniform(&state) * roomSize * 0.8f;  // x
    ballData[1] = 1.0f + curand_uniform(&state) * roomSize * 0.5f;              // y
    ballData[2] = -roomSize * 0.4f + curand_uniform(&state) * roomSize * 0.8f;  // z
    
    // Random radius (0.25x smaller)
    ballData[3] = (0.1f + curand_uniform(&state) * 0.3f) * 1.0f;
    
    // Random color (vibrant)
    ballData[4] = 0.3f + curand_uniform(&state) * 0.7f;  // r
    ballData[5] = 0.3f + curand_uniform(&state) * 0.7f;  // g
    ballData[6] = 0.3f + curand_uniform(&state) * 0.7f;  // b
    
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
    data.vel[idx].x = -2.0f + curand_uniform(&state) * 4.0f;
    data.vel[idx].y = -2.0f + curand_uniform(&state) * 4.0f;
    data.vel[idx].z = -2.0f + curand_uniform(&state) * 4.0f;
    
    // Random angular velocity
    data.angVel[idx].x = -3.0f + curand_uniform(&state) * 6.0f;
    data.angVel[idx].y = -3.0f + curand_uniform(&state) * 6.0f;
    data.angVel[idx].z = -3.0f + curand_uniform(&state) * 6.0f;
    
    // Store radius
    data.radii[idx] = ballData[3];
}

// Host functions callable from BallsDemo.cpp

extern "C" void initCudaPhysics(int numBalls, float roomSize, GLuint vbo, cudaGraphicsResource** vboResource) {
    g_deviceData.numBalls = numBalls;
    g_deviceData.worldOrig = -100.0f;
    
    // Find maximum radius (updated to match smaller balls)
    float maxRadius = (0.1f + 0.3f) * 1.0f;  // Maximum possible radius
    g_deviceData.gridSpacing = 2.5f * maxRadius;
    
    // Allocate physics arrays
    cudaMalloc(&g_deviceData.vel, numBalls * sizeof(Vec3));
    cudaMalloc(&g_deviceData.angVel, numBalls * sizeof(Vec3));
    cudaMalloc(&g_deviceData.prevPos, numBalls * sizeof(Vec3));
    cudaMalloc(&g_deviceData.radii, numBalls * sizeof(float));
    
    // Allocate hash grid
    cudaMalloc(&g_deviceData.hashVals, numBalls * sizeof(int));
    cudaMalloc(&g_deviceData.hashIds, numBalls * sizeof(int));
    cudaMalloc(&g_deviceData.hashCellFirst, HASH_SIZE * sizeof(int));
    cudaMalloc(&g_deviceData.hashCellLast, HASH_SIZE * sizeof(int));
    cudaMalloc(&g_deviceData.sortedPos, numBalls * sizeof(Vec3));
    cudaMalloc(&g_deviceData.sortedPrevPos, numBalls * sizeof(Vec3));
    
    // Initialize memory
    cudaMemset(g_deviceData.vel, 0, numBalls * sizeof(Vec3));
    cudaMemset(g_deviceData.angVel, 0, numBalls * sizeof(Vec3));
    cudaMemset(g_deviceData.prevPos, 0, numBalls * sizeof(Vec3));
    cudaMemset(g_deviceData.hashCellFirst, 0, HASH_SIZE * sizeof(int));
    cudaMemset(g_deviceData.hashCellLast, 0, HASH_SIZE * sizeof(int));
    
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
    kernel_initBalls<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, roomSize, time(nullptr));
    cudaDeviceSynchronize();
    
    cudaGraphicsUnmapResources(1, vboResource, 0);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA initialization error: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void updateCudaPhysics(float dt, Vec3 gravity, float friction, float bounce, float roomSize, 
                                   cudaGraphicsResource* vboResource) {
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
    
    kernel_fillHash<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData);
    
    // Sort by hash
    thrust::device_ptr<int> hashVals(g_deviceData.hashVals);
    thrust::device_ptr<int> hashIds(g_deviceData.hashIds);
    thrust::sort_by_key(hashVals, hashVals + g_deviceData.numBalls, hashIds);
    
    cudaMemset(g_deviceData.hashCellFirst, 0, HASH_SIZE * sizeof(int));
    cudaMemset(g_deviceData.hashCellLast, 0, HASH_SIZE * sizeof(int));
    
    kernel_setupHash<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData);
    
    kernel_ballCollision<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, bounce);
    
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
    
    // Free all device memory
    if (g_deviceData.vel) cudaFree(g_deviceData.vel);
    if (g_deviceData.angVel) cudaFree(g_deviceData.angVel);
    if (g_deviceData.prevPos) cudaFree(g_deviceData.prevPos);
    if (g_deviceData.radii) cudaFree(g_deviceData.radii);
    if (g_deviceData.hashVals) cudaFree(g_deviceData.hashVals);
    if (g_deviceData.hashIds) cudaFree(g_deviceData.hashIds);
    if (g_deviceData.hashCellFirst) cudaFree(g_deviceData.hashCellFirst);
    if (g_deviceData.hashCellLast) cudaFree(g_deviceData.hashCellLast);
    if (g_deviceData.sortedPos) cudaFree(g_deviceData.sortedPos);
    if (g_deviceData.sortedPrevPos) cudaFree(g_deviceData.sortedPrevPos);
    
    // Zero out the structure
    g_deviceData.numBalls = 0;
    g_deviceData.vel = nullptr;
    g_deviceData.angVel = nullptr;
    g_deviceData.prevPos = nullptr;
    g_deviceData.radii = nullptr;
    g_deviceData.hashVals = nullptr;
    g_deviceData.hashIds = nullptr;
    g_deviceData.hashCellFirst = nullptr;
    g_deviceData.hashCellLast = nullptr;
    g_deviceData.sortedPos = nullptr;
    g_deviceData.sortedPrevPos = nullptr;
    g_deviceData.vboData = nullptr;
    
    // Unregister VBO
    if (vboResource) {
        cudaGraphicsUnregisterResource(vboResource);
    }
}

