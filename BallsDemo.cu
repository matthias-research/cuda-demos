#include "BallsDemo.h"
#include "CudaUtils.h"
#include "CudaMeshes.h"
#include "CudaHash.h"
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

static const int VBO_STRIDE = 14;// Interleaved: pos(3), radius(1), color(3), quat(4), pad(3) = 14 floats

struct BallsDeviceData {
    void free()  // no destructor because cuda would call it in the kernels
    {
        numBalls = 0;
        vel.free();
        prevPos.free();
        angVel.free();
        radii.free();
        posCorr.free();
        newAngVel.free();
        raycastHit.free();
    }
    
    size_t allocationSize() 
    {
        size_t s = 0;
        s += vel.allocationSize();
        s += prevPos.allocationSize();
        s += angVel.allocationSize();
        s += radii.allocationSize();
        s += posCorr.allocationSize();
        s += newAngVel.allocationSize();
        return s;
    }
    
    int numBalls = 0;
    int  numMeshes = 0;
    float gridSpacing = 0.0f;
    float worldOrig = 0.0f;
    float maxRadius = 0.0f;
    
    // Physics data
    DeviceBuffer<Vec3> vel;
    DeviceBuffer<Vec3> prevPos;
    DeviceBuffer<Vec3> angVel;
    DeviceBuffer<float> radii;
            
    // Collision correction buffers (for symmetric resolution)
    DeviceBuffer<Vec3> posCorr;
    DeviceBuffer<Vec4> newAngVel;

    // VBO data (mapped from OpenGL)
    float* vboData = nullptr;      // Interleaved: pos(3), radius(1), color(3), quat(4), pad(3) = 14 floats

    DeviceBuffer<float> raycastHit;
};


__global__ void kernel_integrate(BallsDeviceData data, float dt, Vec3 gravity, float friction, float terminalVelocity) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    data.prevPos[idx] = pos;

    Vec3 vel = data.vel[idx];
    vel += gravity * dt;

    float v = vel.length();
    if (v > terminalVelocity)
        vel *= terminalVelocity / v;

    vel *= friction;

    data.vel[idx] = vel;

    pos += vel * dt;
    
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}


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
    if (idx >= otherIdx) 
        return;
    
    Vec3 delta = otherPos - pos;
    float dist = delta.magnitude();
    float minDist = radius + otherRadius;
    
    if (dist < minDist && dist > 0.001f) 
    {
        Vec3 normal = delta / dist;
                float overlap = minDist - dist;
        Vec3 separation = normal * (overlap * 0.5f);
        
        atomicAdd(&data.posCorr[idx].x, -separation.x);
        atomicAdd(&data.posCorr[idx].y, -separation.y);
        atomicAdd(&data.posCorr[idx].z, -separation.z);
        
        atomicAdd(&data.posCorr[otherIdx].x, separation.x);
        atomicAdd(&data.posCorr[otherIdx].y, separation.y);
        atomicAdd(&data.posCorr[otherIdx].z, separation.z);
        
        Vec3 relativeVel = data.vel[otherIdx] - data.vel[idx];
        Vec3 tangentialRelVel = relativeVel - normal * normal.dot(relativeVel);
        
        Vec3 angVelContrib1 = tangentialRelVel.cross(normal) / radius;
        Vec3 angVelContrib2 = tangentialRelVel.cross(-normal) / otherRadius;
        
        atomicAdd(&data.newAngVel[idx].x, angVelContrib1.x);
        atomicAdd(&data.newAngVel[idx].y, angVelContrib1.y);
        atomicAdd(&data.newAngVel[idx].z, angVelContrib1.z);
        atomicAdd(&data.newAngVel[idx].w, 1.0f);
        
        atomicAdd(&data.newAngVel[otherIdx].x, angVelContrib2.x);
        atomicAdd(&data.newAngVel[otherIdx].y, angVelContrib2.y);
        atomicAdd(&data.newAngVel[otherIdx].z, angVelContrib2.z);
        atomicAdd(&data.newAngVel[otherIdx].w, 1.0f);
    }
}

__global__ void kernel_ballCollision_neighbors(BallsDeviceData data, HashDeviceData hashData, float bounce) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];

    int firstNeighbor = hashData.firstNeighbor[idx];
    int numNeighbors = hashData.firstNeighbor[idx + 1] - firstNeighbor;

    for (int i = 0; i < numNeighbors; i++) 
    {
        int otherIdx = hashData.neighbors[firstNeighbor + i];
        float* otherData = data.vboData + otherIdx * VBO_STRIDE;
        Vec3 otherPos(otherData[0], otherData[1], otherData[2]);
        float otherRadius = otherData[3];
        
        handleBallCollision(idx, otherIdx, pos, radius, otherPos, otherRadius, bounce, data);
    }
}


__global__ void kernel_ballCollision_hash(BallsDeviceData data, HashDeviceData hashData, float bounce) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;

    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];

    int xi = floorf((pos.x - data.worldOrig) / data.gridSpacing);
    int yi = floorf((pos.y - data.worldOrig) / data.gridSpacing);
    int zi = floorf((pos.z - data.worldOrig) / data.gridSpacing);

    // Check neighboring cells

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int cellX = xi + dx;
                int cellY = yi + dy;
                int cellZ = zi + dz;

                unsigned int h = hashFunction(cellX, cellY, cellZ);

                int first = hashData.hashCellFirst[h];
                int last = hashData.hashCellLast[h];

                // Check all balls in this cell
                for (int i = first; i < last; i++) {
                    int otherIdx = hashData.hashIds[i];

                    float* otherData = data.vboData + otherIdx * VBO_STRIDE;
                    Vec3 otherPos(otherData[0], otherData[1], otherData[2]);
                    float otherRadius = otherData[3];

                    handleBallCollision(idx, otherIdx, pos, radius, otherPos, otherRadius, bounce, data);
                }
            }
        }
    }
}


__device__ void handlePlaneCollision(Vec3& pos, float radius, Vec3& angVel, const Vec3& vel, const Plane& plane, float restitution, float dt)
{
    float signedDist = plane.n.dot(pos) - radius - plane.d;
    
    if (signedDist < 0) {
        pos -= plane.n * signedDist;
        
        // restitution
        float normalVel = vel.dot(plane.n);

        if (normalVel < 0) { // Only if moving towards the wall
            pos -= normalVel * plane.n * restitution * dt; // Correct position based on velocity
        }
        
        Vec3 tangentialVel = vel - plane.n * plane.n.dot(vel);
        Vec3 newAngVel = tangentialVel.cross(plane.n) / radius;
        angVel = newAngVel;
    }
}

__global__ void kernel_wallCollision(BallsDeviceData data, Bounds3 sceneBounds, float restitution, float dt, bool updateAngularVel) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) 
        return;
        
    Plane walls[5] = {
        Plane(Vec3(-1, 0, 0),  -sceneBounds.maximum.x),
        Plane(Vec3(1, 0, 0),   sceneBounds.minimum.x),
        Plane(Vec3(0, 1, 0),   sceneBounds.minimum.y),
        Plane(Vec3(0, 0, -1),  -sceneBounds.maximum.z),
        Plane(Vec3(0, 0, 1),   sceneBounds.minimum.z)
    };
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    Vec3 angVel = data.angVel[idx];
    
    float radius = ballData[3];

    for (int i = 0; i < 5; i++) {
        const Plane& wall = walls[i];
        handlePlaneCollision(pos, radius, angVel, data.vel[idx], wall, restitution, dt);
    }
    
    data.angVel[idx] = angVel;

    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}

__device__ void kernel_ballMeshCollision(BallsDeviceData& data, const MeshesDeviceData& meshData, int ballIdx, int meshIdx, float searchRadius, float restitution, float dt) 
{
    // traverse the triangle BVH

    float* ballData = data.vboData + ballIdx * VBO_STRIDE;
    Vec3 ballPos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    Bounds3 queryBounds(ballPos - Vec3(searchRadius, searchRadius, searchRadius), 
                       ballPos + Vec3(searchRadius, searchRadius, searchRadius));
    
    int rootNode = meshData.trianglesBvh.mRootNodes[meshIdx];
    if (rootNode < 0) 
        return;
        
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        PackedNodeHalf lower = meshData.trianglesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = meshData.trianglesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z),
            Vec3(upper.x, upper.y, upper.z));

        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf node - contains a triangle
                int triIdx = leftIndex;

                int i0 = meshData.triIds.buffer[triIdx * 3 + 0];
                int i1 = meshData.triIds.buffer[triIdx * 3 + 1];
                int i2 = meshData.triIds.buffer[triIdx * 3 + 2];

                Vec3 v0 = meshData.vertices.buffer[i0];
                Vec3 v1 = meshData.vertices.buffer[i1];
                Vec3 v2 = meshData.vertices.buffer[i2];

                Vec3 baryCoords = getClosestPointOnTriangle(ballPos, v0, v1, v2);
                Vec3 point = v0 * baryCoords.x + v1 * baryCoords.y + v2 * baryCoords.z;

                Vec3 n = (ballPos - point).normalized();
                handlePlaneCollision(ballPos, radius, data.angVel[ballIdx], data.vel[ballIdx], Plane(n, n.dot(point)), restitution, dt);
            }
            else {  // Internal node
                if (stackCount < 63) {  
                    stack[stackCount++] = leftIndex;
                    stack[stackCount++] = rightIndex;
                }
            }
        }
    }
    ballData[0] = ballPos.x;
    ballData[1] = ballPos.y;
    ballData[2] = ballPos.z;
}


__global__ void kernel_ballMeshesCollision(BallsDeviceData data, const MeshesDeviceData meshData, float searchRadius, float restitution, float dt)
{
    // traverse the mesh BVH

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    if (meshData.numMeshTriangles == 0) 
        return;
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 ballPos(ballData[0], ballData[1], ballData[2]);
    
    Bounds3 queryBounds(ballPos - Vec3(searchRadius, searchRadius, searchRadius), 
                       ballPos + Vec3(searchRadius, searchRadius, searchRadius));
    
    int rootNode = meshData.meshesBvh.mRootNodes[0];
    if (rootNode < 0) return;
        
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        PackedNodeHalf lower = meshData.meshesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = meshData.meshesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z),
            Vec3(upper.x, upper.y, upper.z));

        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf node - contains a mesh
                int meshIdx = leftIndex;
                kernel_ballMeshCollision(data, meshData, idx, meshIdx, searchRadius, restitution, dt);
            }

            else {  // Internal node
                if (stackCount < 63) {  // Prevent stack overflow
                    stack[stackCount++] = leftIndex;
                    stack[stackCount++] = rightIndex;
                }
            }
        }
    }
}

// Apply collision corrections with relaxation (PBD)
__global__ void kernel_applyCorrections(BallsDeviceData data, float relaxation)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) 
        return;
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    pos += data.posCorr[idx] * relaxation;
    
    Vec4 angVelAccum = data.newAngVel[idx];

    if (angVelAccum.w > 0.0f) 
    { 
        Vec3 avgAngVel(angVelAccum.x / angVelAccum.w, 
                       angVelAccum.y / angVelAccum.w, 
                       angVelAccum.z / angVelAccum.w);
        data.angVel[idx] = avgAngVel;
    }
    
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}


__global__ void kernel_deriveVelocity(BallsDeviceData data, float dt) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    data.vel[idx] = (pos - data.prevPos[idx]) / dt;
}


__global__ void kernel_integrateQuaternions(BallsDeviceData data, float dt) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    
    Quat quat(ballData[7], ballData[8], ballData[9], ballData[10]); 
    Vec3 omega = data.angVel[idx] * dt;
    quat = quat.rotateLinear(quat, omega);
    
    ballData[7] = quat.x;
    ballData[8] = quat.y;
    ballData[9] = quat.z;
    ballData[10] = quat.w;
}


__global__ void kernel_initBalls(BallsDeviceData data, Bounds3 ballsBounds, float minRadius, float maxRadius,
                                  int ballsPerLayer, int ballsPerRow, int ballsPerCol, float gridSpacing, unsigned long seed) 
{
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
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    
    // Grid-based position within ballsBounds
    int layer = idx / ballsPerLayer;
    int inLayer = idx % ballsPerLayer;
    int row = inLayer / ballsPerRow;
    int col = inLayer % ballsPerRow;
    
    ballData[0] = ballsBounds.minimum.x + gridSpacing + col * gridSpacing;  // x
    ballData[2] = ballsBounds.minimum.z + gridSpacing + row * gridSpacing;  // z
    ballData[1] = ballsBounds.minimum.y + gridSpacing + layer * gridSpacing; // y
    
    // Random radius
    ballData[3] = minRadius + FRAND() * (maxRadius - minRadius);
    
    // Beach ball colors: red, blue, yellow, white, orange, cyan
    float colorChoice = FRAND();
    if (colorChoice < 0.16f) {
        ballData[4] = 0.9f; ballData[5] = 0.1f; ballData[6] = 0.1f;
    } else if (colorChoice < 0.32f) {
        ballData[4] = 0.1f; ballData[5] = 0.5f; ballData[6] = 0.95f;
    } else if (colorChoice < 0.48f) {
        ballData[4] = 0.95f; ballData[5] = 0.85f; ballData[6] = 0.1f;
    } else if (colorChoice < 0.64f) {
        ballData[4] = 0.95f; ballData[5] = 0.95f; ballData[6] = 0.95f;
    } else if (colorChoice < 0.80f) {
        ballData[4] = 0.95f; ballData[5] = 0.5f; ballData[6] = 0.1f;
    } else {
        ballData[4] = 0.1f; ballData[5] = 0.85f; ballData[6] = 0.85f;
    }
    
    // Random orientation quaternion (random axis-angle rotation)
    float angle = FRAND() * 3.14159f * 2.0f;  // Random angle [0, 2π]
    float axisMag = sqrtf(FRAND());  // Magnitude for random unit vector
    float axisTheta = FRAND() * 3.14159f * 2.0f;
    float axisPhi = acosf(2.0f * FRAND() - 1.0f);
    
    // Random unit axis
    float axisX = sinf(axisPhi) * cosf(axisTheta);
    float axisY = sinf(axisPhi) * sinf(axisTheta);
    float axisZ = cosf(axisPhi);
    
    // Convert axis-angle to quaternion
    float halfAngle = angle * 0.5f;
    float sinHalfAngle = sinf(halfAngle);
    ballData[7] = cosf(halfAngle);  // w
    ballData[8] = axisX * sinHalfAngle;  // x
    ballData[9] = axisY * sinHalfAngle;  // y
    ballData[10] = axisZ * sinHalfAngle; // z
    
    // Padding
    ballData[11] = 0.0f;
    ballData[12] = 0.0f;
    ballData[13] = 0.0f;
    
    // Random velocity
    data.vel[idx].x = -2.0f + FRAND() * 4.0f;
    data.vel[idx].y = -2.0f + FRAND() * 4.0f;
    data.vel[idx].z = -2.0f + FRAND() * 4.0f;
    
    // Zero initial angular velocity (will be set by collisions)
    data.angVel[idx].x = 0.0f;
    data.angVel[idx].y = 0.0f;
    data.angVel[idx].z = 0.0f;
    
    // Store radius
    data.radii[idx] = ballData[3];
    
    #undef FRAND
}


//-----------------------------------------------------------------------------

// Host functions callable from BallsDemo.cpp

bool BallsDemo::cudaRaycast(const Ray& ray, float& minT)
{
    minT = MaxFloat;

    if (!meshes)
        return false;

    deviceData->raycastHit.resize(1);

    meshes->rayCast(1, nullptr, ray, deviceData->raycastHit.buffer, 1);
    deviceData->raycastHit.getDeviceObject(minT, 0);
    return minT < MaxFloat;
}


void BallsDemo::initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, Scene* scene) 
{
    if (!deviceData) 
        deviceData = std::make_shared<BallsDeviceData>();
    if (!meshes) 
        meshes = std::make_shared<CudaMeshes>();
    if (!hash) 
        hash = std::make_shared<CudaHash>();

    meshes->initialize(scene);
    hash->initialize();
    
    deviceData->numBalls = demoDesc.numBalls;
    deviceData->worldOrig = -100.0f;
    
    deviceData->maxRadius = demoDesc.maxRadius;
    deviceData->gridSpacing = 2.5f * demoDesc.maxRadius;
    
    deviceData->vel.resize(demoDesc.numBalls, false);
    deviceData->prevPos.resize(demoDesc.numBalls, false);
    deviceData->angVel.resize(demoDesc.numBalls, false);
    deviceData->radii.resize(demoDesc.numBalls, false);
            deviceData->posCorr.resize(demoDesc.numBalls, false);
    deviceData->newAngVel.resize(demoDesc.numBalls, false);
    
    deviceData->vel.setZero();
    deviceData->angVel.setZero();

    // Cuda - OpenGL interop
    cudaGraphicsGLRegisterBuffer(vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, *vboResource);
    
    deviceData->vboData = d_vboData;
    
    // Calculate grid layout parameters based on ballsBounds
    float gridSpacing = 2.0f * demoDesc.maxRadius;  // Space between ball centers
    Vec3 spawnSize = demoDesc.ballsBounds.maximum - demoDesc.ballsBounds.minimum;
    float usableWidth = spawnSize.x - 2.0f * gridSpacing;  // Leave margin
    float usableDepth = spawnSize.z - 2.0f * gridSpacing;
    int ballsPerRow = (int)(usableWidth / gridSpacing);
    if (ballsPerRow < 1) ballsPerRow = 1;
    int ballsPerCol = (int)(usableDepth / gridSpacing);
    if (ballsPerCol < 1) ballsPerCol = 1;
    int ballsPerLayer = ballsPerRow * ballsPerCol;
    
    // Initialize balls with grid data
    int numBlocks = (demoDesc.numBalls + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching kernel_initBalls with %d blocks, %d threads per block, %d total balls\n", numBlocks, THREADS_PER_BLOCK, demoDesc.numBalls);
    printf("Grid layout: %d balls per row, %d balls per layer, spacing: %.2f\n", ballsPerRow, ballsPerLayer, gridSpacing);
    
    kernel_initBalls<<<numBlocks, THREADS_PER_BLOCK>>>(*deviceData, demoDesc.ballsBounds, demoDesc.minRadius, demoDesc.maxRadius,
                                                        ballsPerLayer, ballsPerRow, ballsPerCol, gridSpacing, (unsigned long)time(nullptr));
    
    cudaGraphicsUnmapResources(1, vboResource, 0);
}


void BallsDemo::updateCudaPhysics(float dt, cudaGraphicsResource* vboResource) 
{
    if (deviceData->numBalls == 0) 
        return;

    const bool useNeighbors = false;

    int numBlocks = (deviceData->numBalls + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Cuda - OpenGL interop

    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, &vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, vboResource);
    
    deviceData->vboData = d_vboData;

    // Recompute hash - outside the substep loop

    hash->fillHash(deviceData->numBalls, deviceData->vboData, VBO_STRIDE, deviceData->gridSpacing);

    if (useNeighbors)
        hash->findNeighbors(deviceData->numBalls, deviceData->vboData, VBO_STRIDE);


    int numSubSteps = 5;
    float sdt = dt / (float)numSubSteps;

    auto meshesData = meshes->getDeviceData();
    auto hashData = hash->getDeviceData();

    for (int subStep = 0; subStep < numSubSteps; subStep++)
    {
        // Integrate

        kernel_integrate << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, sdt, Vec3(0.0f, -demoDesc.gravity, 0.0f), demoDesc.friction, demoDesc.terminalVelocity);

        deviceData->posCorr.setZero();
        deviceData->newAngVel.setZero();

        // Inter ball collision detection

        if (useNeighbors)
            kernel_ballCollision_neighbors << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, *hashData, demoDesc.bounce);
        else
            kernel_ballCollision_hash << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, *hashData, demoDesc.bounce);
        cudaCheck(cudaGetLastError());

        const float relaxation = 0.3f;
        kernel_applyCorrections << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, relaxation);

        kernel_wallCollision << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, demoDesc.sceneBounds, demoDesc.bounce, sdt, subStep == 0);

        // Ball-mesh collision

        if (meshesData->numMeshTriangles > 0) 
        {
            float meshSearchRadius = 1.0f * deviceData->maxRadius;  // Search radius for mesh collisions
            kernel_ballMeshesCollision << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, *meshesData, meshSearchRadius, demoDesc.bounce, sdt);
        }

        // Derive velocity from position change (PBD)

        kernel_deriveVelocity << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, sdt);
        kernel_integrateQuaternions << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, sdt);
    }

    // compute ball shadow depths (writes directly to VBO at offset 11)

    if (meshes)
    {
        Ray sunRay(Vec3(Zero), demoDesc.sunDirection);
        meshes->rayCast(deviceData->numBalls, deviceData->vboData, sunRay, deviceData->vboData + 11, VBO_STRIDE);
    }

    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &vboResource, 0);
    cudaCheck(cudaGetLastError());
}

void BallsDemo::cleanupCudaPhysics(cudaGraphicsResource* vboResource) 
{
    if (deviceData->numBalls == 0) 
        return;
    
    
    if (vboResource) {
        cudaGraphicsUnmapResources(1, &vboResource, 0);
    }
    
    if (deviceData) 
    {
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

void BallsDemo::exportBallsToFile(const std::string& filename)
{
    if (!deviceData || deviceData->numBalls == 0) {
        std::cerr << "No balls to export\n";
        return;
    }

    int numBalls = deviceData->numBalls;
    
    std::vector<Vec3> hostPositions;
    std::vector<float> hostRadii;
    deviceData->prevPos.get(hostPositions, numBalls);
    deviceData->radii.get(hostRadii, numBalls);
    
    FILE* file;
    errno_t err = fopen_s(&file, filename.c_str(), "wb");
    if (err != 0 || !file) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    
    fwrite(&numBalls, sizeof(int32_t), 1, file);
    
    for (int i = 0; i < numBalls; i++) {
        fwrite(&hostPositions[i].x, sizeof(float), 1, file);
        fwrite(&hostPositions[i].y, sizeof(float), 1, file);
        fwrite(&hostPositions[i].z, sizeof(float), 1, file);
        fwrite(&hostRadii[i], sizeof(float), 1, file);
    }
    
    fclose(file);
    
    std::cout << "Exported " << numBalls << " balls to " << filename << "\n";
}


