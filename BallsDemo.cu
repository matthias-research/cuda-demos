#include "BallsDemo.h"
#include "CudaUtils.h"
#include "CudaMeshes.h"
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

// Hash grid parameters
static const int HASH_SIZE = 37000111;  // Prime number for better distribution

static const int VBO_STRIDE = 14;// Interleaved: pos(3), radius(1), color(3), quat(4), pad(3) = 14 floats

// Device data structure - all simulation state on GPU
struct BallsDeviceData {
    void free()  // no destructor because cuda would call it in the kernels
    {
        numBalls = 0;
        vel.free();
        prevPos.free();
        angVel.free();
        radii.free();
        hashVals.free();
        hashIds.free();
        hashCellFirst.free();
        hashCellLast.free();
        posCorr.free();
        newAngVel.free();
        raycastHit.free();
    }
    
    size_t allocationSize() const
    {
        size_t s = 0;
        s += vel.allocationSize();
        s += prevPos.allocationSize();
        s += angVel.allocationSize();
        s += radii.allocationSize();
        s += hashVals.allocationSize();
        s += hashIds.allocationSize();
        s += hashCellFirst.allocationSize();
        s += hashCellLast.allocationSize();
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
    DeviceBuffer<Vec3> vel;           // Velocities
    DeviceBuffer<Vec3> prevPos;       // Previous positions (for PBD)
    DeviceBuffer<Vec3> angVel;        // Angular velocities
    DeviceBuffer<float> radii;        // Radius per ball (for variable sizes)
    
    // Hash grid
    DeviceBuffer<int> hashVals;       // Hash value per ball
    DeviceBuffer<int> hashIds;        // Ball index (gets sorted)
    DeviceBuffer<int> hashCellFirst;  // First ball in each cell
    DeviceBuffer<int> hashCellLast;   // Last ball in each cell
        
    // Collision correction buffers (for symmetric resolution)
    DeviceBuffer<Vec3> posCorr;       // Position corrections
    DeviceBuffer<Vec4> newAngVel;     // New angular velocity accumulation (x,y,z = sum, w = count)

    // VBO data (mapped from OpenGL)
    float* vboData = nullptr;      // Interleaved: pos(3), radius(1), color(3), quat(4), pad(3) = 14 floats

    DeviceBuffer<float> raycastHit;
};

// Factory functions for BallsDeviceData
BallsDeviceData* createBallsDeviceData() {
    return new BallsDeviceData();
}

void deleteBallsDeviceData(BallsDeviceData* data) {
    if (data) {
        delete data;
    }
}

// Helper device functions
__device__ inline int hashPosition(const Vec3& pos, float gridSpacing, float worldOrig) {
    int xi = floorf((pos.x - worldOrig) / gridSpacing);
    int yi = floorf((pos.y - worldOrig) / gridSpacing);
    int zi = floorf((pos.z - worldOrig) / gridSpacing);
    
    unsigned int h = abs((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)) % HASH_SIZE;
    return h;
}

// Integrate - save previous position and predict new position (PBD)
__global__ void kernel_integrate(BallsDeviceData data, float dt, Vec3 gravity, float friction, float terminalVelocity) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read from VBO
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Store previous position
    data.prevPos[idx] = pos;
    
    // Update velocity
    Vec3 vel = data.vel[idx];

    vel += gravity * dt;

    float v = vel.length();
    if (v > terminalVelocity)
        vel *= terminalVelocity / v;

    vel *= friction;

    data.vel[idx] = vel;

    pos += vel * dt;
    
    // Write predicted position back to VBO
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}


// Fill hash values
__global__ void kernel_fillHash(BallsDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read position from VBO
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Compute hash
    int h = hashPosition(pos, data.gridSpacing, data.worldOrig);
    
    data.hashVals[idx] = h;
    data.hashIds[idx] = idx;
}

// Setup hash grid cell boundaries
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
        
        // Calculate angular velocity for ball collisions using tangential relative velocity
        Vec3 relativeVel = data.vel[otherIdx] - data.vel[idx];
        
        // Extract tangential component: v_tangential = v_rel - (v_rel · n) * n
        Vec3 tangentialRelVel = relativeVel - normal * normal.dot(relativeVel);
        
        // No-slip condition for ball collision: ω = (v_tangential × n) / r
        // Each ball gets angular velocity based on its own radius
        Vec3 angVelContrib1 = tangentialRelVel.cross(normal) / radius;
        Vec3 angVelContrib2 = tangentialRelVel.cross(-normal) / otherRadius;
        
        // Accumulate angular velocity contributions atomically
        atomicAdd(&data.newAngVel[idx].x, angVelContrib1.x);
        atomicAdd(&data.newAngVel[idx].y, angVelContrib1.y);
        atomicAdd(&data.newAngVel[idx].z, angVelContrib1.z);
        atomicAdd(&data.newAngVel[idx].w, 1.0f);  // Increment collision count
        
        atomicAdd(&data.newAngVel[otherIdx].x, angVelContrib2.x);
        atomicAdd(&data.newAngVel[otherIdx].y, angVelContrib2.y);
        atomicAdd(&data.newAngVel[otherIdx].z, angVelContrib2.z);
        atomicAdd(&data.newAngVel[otherIdx].w, 1.0f);  // Increment collision count
    }
}

// Ball-to-ball collision (hash grid)
__global__ void kernel_ballCollision(BallsDeviceData data, float bounce) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Get ball data (read from VBO)
    float* ballData = data.vboData + idx * VBO_STRIDE;
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
                    float* otherData = data.vboData + otherIdx * VBO_STRIDE;
                    Vec3 otherPos(otherData[0], otherData[1], otherData[2]);
                    float otherRadius = otherData[3];
                    
                    // Handle collision using symmetric resolution
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

// Wall collision (generalized for arbitrary planes)
__global__ void kernel_wallCollision(BallsDeviceData data, Bounds3 sceneBounds, float restitution, float dt, bool updateAngularVel) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) 
        return;
        
    // Define 5 walls as planes (normal points inward, n·p + d = 0 on plane)
    // Plane equation: n·p + d = 0, where positive values of (n·p + d) mean "inside the room"
    // No ceiling - open upwards
    Plane walls[5] = {
        Plane(Vec3(-1, 0, 0),  -sceneBounds.maximum.x),    // +X wall (normal points left)
        Plane(Vec3(1, 0, 0),   sceneBounds.minimum.x),   // -X wall (normal points right)
        Plane(Vec3(0, 1, 0),   sceneBounds.minimum.y),   // Floor (normal points up)
        Plane(Vec3(0, 0, -1),  -sceneBounds.maximum.z),    // +Z wall (normal points forward)
        Plane(Vec3(0, 0, 1),   sceneBounds.minimum.z)    // -Z wall (normal points back)
    };
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    Vec3 angVel = data.angVel[idx];
    
    float radius = ballData[3];

    // Check collision with each wall
    for (int i = 0; i < 5; i++) {
        const Plane& wall = walls[i];

        handlePlaneCollision(pos, radius, angVel, data.vel[idx], wall, restitution, dt);
    }
    
    // Write back corrected position
    data.angVel[idx] = angVel;

    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}

// BVH-based ball collision (alternative to hash grid)
// Ball-mesh collision using BVH
__device__ void kernel_ballMeshCollision(BallsDeviceData& data, const MeshesDeviceData& meshData, int ballIdx, int meshIdx, float searchRadius, float restitution, float dt) {

    // Get ball data (read from VBO)
    float* ballData = data.vboData + ballIdx * VBO_STRIDE;
    Vec3 ballPos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    // Query bounds (ball position + search radius)
    Bounds3 queryBounds(ballPos - Vec3(searchRadius, searchRadius, searchRadius), 
                       ballPos + Vec3(searchRadius, searchRadius, searchRadius));
    
    // Get root node of triangle BVH
    int rootNode = meshData.trianglesBvh.mRootNodes[meshIdx];
    if (rootNode < 0) 
        return;
        
    // Stack-based BVH traversal
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        // Get node bounds
        PackedNodeHalf lower = meshData.trianglesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = meshData.trianglesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z),
            Vec3(upper.x, upper.y, upper.z));

        // Test intersection with query bounds
        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf node - contains a triangle
                int triIdx = leftIndex;

                // Get the three vertex indices for this triangle
                int i0 = meshData.triIds.buffer[triIdx * 3 + 0];
                int i1 = meshData.triIds.buffer[triIdx * 3 + 1];
                int i2 = meshData.triIds.buffer[triIdx * 3 + 2];

                // Get the three vertices
                Vec3 v0 = meshData.vertices.buffer[i0];
                Vec3 v1 = meshData.vertices.buffer[i1];
                Vec3 v2 = meshData.vertices.buffer[i2];

                // Compute closest point on triangle to ball center
                Vec3 baryCoords = getClosestPointOnTriangle(ballPos, v0, v1, v2);
                Vec3 point = v0 * baryCoords.x + v1 * baryCoords.y + v2 * baryCoords.z;

                Vec3 n = (ballPos - point).normalized();
                handlePlaneCollision(ballPos, radius, data.angVel[ballIdx], data.vel[ballIdx], Plane(n, n.dot(point)), restitution, dt);
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
    // Write corrected position back to VBO
    ballData[0] = ballPos.x;
    ballData[1] = ballPos.y;
    ballData[2] = ballPos.z;
}

// Ball-meshes collision using BVH
__global__ void kernel_ballMeshesCollision(BallsDeviceData data, const MeshesDeviceData meshData, float searchRadius, float restitution, float dt)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Check if we have mesh data
    if (meshData.numMeshTriangles == 0) return;
    
    // Get ball data (read from VBO)
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 ballPos(ballData[0], ballData[1], ballData[2]);
    
    // Query bounds (ball position + search radius)
    Bounds3 queryBounds(ballPos - Vec3(searchRadius, searchRadius, searchRadius), 
                       ballPos + Vec3(searchRadius, searchRadius, searchRadius));
    
    // Get root node of meshes BVH
    int rootNode = meshData.meshesBvh.mRootNodes[0];
    if (rootNode < 0) return;
        
    // Stack-based BVH traversal
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];

        // Get node bounds
        PackedNodeHalf lower = meshData.meshesBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = meshData.meshesBvh.mNodeUppers[nodeIndex];

        Bounds3 nodeBounds(Vec3(lower.x, lower.y, lower.z),
            Vec3(upper.x, upper.y, upper.z));

        // Test intersection with query bounds
        if (nodeBounds.intersect(queryBounds)) {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b) {  // Leaf node - contains a mesh
                int meshIdx = leftIndex;
                kernel_ballMeshCollision(data, meshData, idx, meshIdx, searchRadius, restitution, dt);
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

// Apply collision corrections with relaxation (PBD)
__global__ void kernel_applyCorrections(BallsDeviceData data, float relaxation) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read from VBO
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Apply position corrections with relaxation factor
    pos += data.posCorr[idx] * relaxation;
    
    // Apply angular velocity corrections with averaging
    Vec4 angVelAccum = data.newAngVel[idx];
    if (angVelAccum.w > 0.0f) {  // If there were collisions
        // Average the accumulated angular velocity
        Vec3 avgAngVel(angVelAccum.x / angVelAccum.w, 
                       angVelAccum.y / angVelAccum.w, 
                       angVelAccum.z / angVelAccum.w);
        data.angVel[idx] = avgAngVel;
    }
    
    // Write position back to VBO
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}

// Derive velocity from position change (PBD)
__global__ void kernel_deriveVelocity(BallsDeviceData data, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read current position from VBO
    float* ballData = data.vboData + idx * VBO_STRIDE;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Derive velocity: v = (pos - prevPos) / dt
    data.vel[idx] = (pos - data.prevPos[idx]) / dt;
}

//  Integrate quaternions
__global__ void kernel_integrateQuaternions(BallsDeviceData data, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * VBO_STRIDE;
    
    // Read quaternion
    Quat quat(ballData[7], ballData[8], ballData[9], ballData[10]); // x, y, z, w
    
    // Integrate rotation
    Vec3 omega = data.angVel[idx] * dt;
    quat = quat.rotateLinear(quat, omega);
    
    // Write back
    ballData[7] = quat.x;
    ballData[8] = quat.y;
    ballData[9] = quat.z;
    ballData[10] = quat.w;
}

// Initialization kernel - sets up balls on a grid
__global__ void kernel_initBalls(BallsDeviceData data, Bounds3 ballsBounds, float minRadius, float maxRadius,
                                  int ballsPerLayer, int ballsPerRow, int ballsPerCol, float gridSpacing, unsigned long seed) {
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
        // Red
        ballData[4] = 0.9f; ballData[5] = 0.1f; ballData[6] = 0.1f;
    } else if (colorChoice < 0.32f) {
        // Blue
        ballData[4] = 0.1f; ballData[5] = 0.5f; ballData[6] = 0.95f;
    } else if (colorChoice < 0.48f) {
        // Yellow
        ballData[4] = 0.95f; ballData[5] = 0.85f; ballData[6] = 0.1f;
    } else if (colorChoice < 0.64f) {
        // White
        ballData[4] = 0.95f; ballData[5] = 0.95f; ballData[6] = 0.95f;
    } else if (colorChoice < 0.80f) {
        // Orange
        ballData[4] = 0.95f; ballData[5] = 0.5f; ballData[6] = 0.1f;
    } else {
        // Cyan
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

    return meshes->rayCast(1, nullptr, ray, deviceData->raycastHit.buffer, 1);
    deviceData->raycastHit.getDeviceObject(minT, 0);
    return minT < MaxFloat;
}


void BallsDemo::initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, Scene* scene) {
    // Ensure any previous CUDA operations are complete
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before init: %s\n", cudaGetErrorString(err));
    }

    if (!deviceData) 
        deviceData = std::make_shared<BallsDeviceData>();
    if (!meshes) 
        meshes = std::make_shared<CudaMeshes>();

    meshes->initialize(scene);
    
    deviceData->numBalls = demoDesc.numBalls;
    deviceData->worldOrig = -100.0f;
    
    // Use provided maximum radius
    deviceData->maxRadius = demoDesc.maxRadius;
    deviceData->gridSpacing = 2.5f * demoDesc.maxRadius;
    
    // Allocate physics arrays using DeviceBuffer
    deviceData->vel.resize(demoDesc.numBalls, false);
    deviceData->prevPos.resize(demoDesc.numBalls, false);
    deviceData->angVel.resize(demoDesc.numBalls, false);
    deviceData->radii.resize(demoDesc.numBalls, false);
    
    // Allocate hash grid using DeviceBuffer
    deviceData->hashVals.resize(demoDesc.numBalls, false);
    deviceData->hashIds.resize(demoDesc.numBalls, false);
    deviceData->hashCellFirst.resize(HASH_SIZE, false);
    deviceData->hashCellLast.resize(HASH_SIZE, false);
    
    // Allocate collision correction buffers
    deviceData->posCorr.resize(demoDesc.numBalls, false);
    deviceData->newAngVel.resize(demoDesc.numBalls, false);
    
    // Initialize memory
    deviceData->vel.setZero();
    deviceData->angVel.setZero();
    deviceData->hashCellFirst.setZero();
    deviceData->hashCellLast.setZero();
    
    // Register VBO with CUDA
    cudaGraphicsGLRegisterBuffer(vboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    
    // Map VBO and initialize balls on GPU
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, *vboResource);
    
    // Update VBO pointer
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
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA initialization error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize to ensure initialization completes
    cudaDeviceSynchronize();
}

void BallsDemo::updateCudaPhysics(float dt,
                                   cudaGraphicsResource* vboResource) {
    if (deviceData->numBalls == 0) return;
    
    int numBlocks = (deviceData->numBalls + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Map VBO
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, &vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, vboResource);
    
    // Update VBO pointer
    deviceData->vboData = d_vboData;
    
    // Hash grid collision detection
    kernel_fillHash << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData);

    // Sort by hash
    thrust::device_ptr<int> hashVals(deviceData->hashVals.buffer);
    thrust::device_ptr<int> hashIds(deviceData->hashIds.buffer);
    thrust::sort_by_key(hashVals, hashVals + deviceData->numBalls, hashIds);

    deviceData->hashCellFirst.setZero();
    deviceData->hashCellLast.setZero();

    kernel_setupHash << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData);

    int numSubSteps = 5;

    float sdt = dt / (float)numSubSteps;

    auto meshesData = meshes->getDeviceData();

    for (int subStep = 0; subStep < numSubSteps; subStep++)
    {

        // Run physics pipeline
        kernel_integrate << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, sdt, Vec3(0.0f, -demoDesc.gravity, 0.0f), demoDesc.friction, demoDesc.terminalVelocity);

        // Zero correction buffers before collision detection
        deviceData->posCorr.setZero();
        deviceData->newAngVel.setZero();

        // Hash grid collision detection
        kernel_ballCollision << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, demoDesc.bounce);

        // Apply corrections with relaxation
        const float relaxation = 0.3f;
        kernel_applyCorrections << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, relaxation);

        kernel_wallCollision << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, demoDesc.sceneBounds, demoDesc.bounce, sdt, subStep == 0);

        // Ball-mesh collision
        if (meshesData->numMeshTriangles > 0) {
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

    // Ensure all CUDA operations complete before unmapping VBO
    cudaDeviceSynchronize();

    // Unmap VBO
    cudaGraphicsUnmapResources(1, &vboResource, 0);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA physics error: %s\n", cudaGetErrorString(err));
    }
}

void BallsDemo::cleanupCudaPhysics(cudaGraphicsResource* vboResource) {
    if (deviceData->numBalls == 0) return;
    
    // Synchronize CUDA to ensure all operations complete
    cudaDeviceSynchronize();
    
    // Unmap VBO before cleanup if it's mapped
    if (vboResource) {
        cudaGraphicsUnmapResources(1, &vboResource, 0);
    }
    
    // Free all device memory using DeviceBuffer::free()
    if (deviceData) {

        deviceData->vboData = nullptr;
        deviceData->free();
        deviceData.reset();
    }
    if (meshes) {
        meshes->cleanup();
    }
    
    // Unregister VBO
    if (vboResource) {
        cudaGraphicsUnregisterResource(vboResource);
    }
    
    // Final sync to ensure cleanup completes
    cudaDeviceSynchronize();
}

void BallsDemo::exportBallsToFile(const std::string& filename)
{
    if (!deviceData || deviceData->numBalls == 0) {
        std::cerr << "No balls to export\n";
        return;
    }

    int numBalls = deviceData->numBalls;
    
    // Copy positions from GPU to CPU using vectors
    std::vector<Vec3> hostPositions;
    std::vector<float> hostRadii;
    deviceData->prevPos.get(hostPositions, numBalls);
    deviceData->radii.get(hostRadii, numBalls);
    
    // Write binary file
    FILE* file;
    errno_t err = fopen_s(&file, filename.c_str(), "wb");
    if (err != 0 || !file) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    
    // Write number of balls
    fwrite(&numBalls, sizeof(int32_t), 1, file);
    
    // Write position and radius for each ball
    for (int i = 0; i < numBalls; i++) {
        fwrite(&hostPositions[i].x, sizeof(float), 1, file);
        fwrite(&hostPositions[i].y, sizeof(float), 1, file);
        fwrite(&hostPositions[i].z, sizeof(float), 1, file);
        fwrite(&hostRadii[i], sizeof(float), 1, file);
    }
    
    fclose(file);
    
    std::cout << "Exported " << numBalls << " balls to " << filename << "\n";
}


