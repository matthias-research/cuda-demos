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
#include "CudaMeshes.h"


static const int HASH_SIZE = 37000111;  // Prime number for better distribution
static const int VBO_STRIDE = 6;    // pos(3), color(3)

// Device data structure - all simulation state on GPU
struct FluidDeviceData {
    void free()  // no destructor because cuda would call it in the kernels
    {
        numParticles = 0;

        vel.free();
        prevPos.free();
        hashVals.free();
        hashIds.free();
        hashCellFirst.free();
        hashCellLast.free();
        posCorr.free();
        raycastHit.free();
    }
        
    int numParticles = 0;
    float particleRadius = 0.2f;
    float kernelRadius = 0.4f;
    float gridSpacing = 0.4f;
    float worldOrig = -100.0f;
    
    // Physics data
    DeviceBuffer<Vec3> vel;           // Velocities
    DeviceBuffer<Vec3> prevPos;       // Previous positions (for PBD)
    
    // Hash grid
    DeviceBuffer<int> hashVals;       // Hash value per particle
    DeviceBuffer<int> hashIds;        // Particle index (gets sorted)
    DeviceBuffer<int> hashCellFirst;  // First particle in each cell
    DeviceBuffer<int> hashCellLast;   // Last particle in each cell
        
    // Collision correction buffers (for symmetric resolution)
    DeviceBuffer<Vec3> posCorr;       // Position corrections
    float* vboData = nullptr;      // Interleaved: pos(3), color(3)

    DeviceBuffer<float> raycastHit;
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


// Host functions callable from FluidDemo.cpp

bool FluidDemo::cudaRaycast(const Ray& ray, float& minT)
{
    minT = MaxFloat;

    if (!meshes) 
        return false;

    deviceData->raycastHit.resize(1);

    return meshes->rayCast(1, nullptr, ray, deviceData->raycastHit.buffer, 1);
    deviceData->raycastHit.getDeviceObject(minT, 0);
    return minT < MaxFloat;
}


void FluidDemo::initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, Scene* scene) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before init: %s\n", cudaGetErrorString(err));
    }

    if (!deviceData) 
        deviceData = std::make_shared<FluidDeviceData>();

    if (!meshes) 
        meshes = std::make_shared<CudaMeshes>();

    meshes->initialize(scene);

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
    if (meshes) {
        meshes->cleanup();
    }
    if (vboResource) {
        cudaGraphicsUnregisterResource(vboResource);
    }
    cudaDeviceSynchronize();
}
