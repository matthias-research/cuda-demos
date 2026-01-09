#include "BallsDemo.h"
#include "CudaUtils.h"
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

// Device data structure - all simulation state on GPU
struct BallsDeviceData {
    void free()  // no destructor because cuda would call it in the kernels
    {
        vel.free();
        prevPos.free();
        angVel.free();
        radii.free();
        hashVals.free();
        hashIds.free();
        hashCellFirst.free();
        hashCellLast.free();
        ballBoundsLowers.free();
        ballBoundsUppers.free();
        posCorr.free();
        newAngVel.free();
        meshFirstTriangle.free();
        meshVertices.free();
        meshTriIds.free();
        meshBoundsLower.free();
        meshBoundsUpper.free();
        meshTriBoundsLower.free();
        meshTriBoundsUpper.free();
        numBalls = 0;
        numMeshTriangles = 0;
        meshesBvh.free();
        trianglesBvh.free();
        rayCastMinT.free();
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
        s += ballBoundsLowers.allocationSize();
        s += ballBoundsUppers.allocationSize();
        s += posCorr.allocationSize();
        s += newAngVel.allocationSize();
        s += meshFirstTriangle.allocationSize();
        s += meshVertices.allocationSize();
        s += meshTriIds.allocationSize();
        s += meshBoundsLower.allocationSize();
        s += meshBoundsUpper.allocationSize();
        s += meshTriBoundsLower.allocationSize();
        s += meshTriBoundsUpper.allocationSize();
        s += rayCastMinT.allocationSize();
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
    
    // BVH collision detection
    DeviceBuffer<Vec4> ballBoundsLowers;  // Lower bounds for BVH
    DeviceBuffer<Vec4> ballBoundsUppers;  // Upper bounds for BVH
    BVH ballsBvh;                         // BVH structure for balls
    
    // Mesh collision data
    DeviceBuffer<int> meshFirstTriangle; // Starting triangle index per mesh
    DeviceBuffer<Vec3> meshVertices;      // Concatenated vertices from all scene meshes
    DeviceBuffer<int> meshTriIds;         // Triangle indices (groups of 3)
    int numMeshTriangles = 0;             // Total number of triangles
    
    // Triangle BVH
    DeviceBuffer<Vec4> meshTriBoundsLower; // Lower bounds for each triangle
    DeviceBuffer<Vec4> meshTriBoundsUpper; // Upper bounds for each triangle
    DeviceBuffer<Vec4> meshBoundsLower;   // Overall mesh bounds
    DeviceBuffer<Vec4> meshBoundsUpper;   // Overall mesh bounds
    BVH meshesBvh;                        // BVH structure for entire meshes
    BVH trianglesBvh;                      // Separate BVH for triangles
    
    // Collision correction buffers (for symmetric resolution)
    DeviceBuffer<Vec3> posCorr;       // Position corrections
    DeviceBuffer<Vec4> newAngVel;     // New angular velocity accumulation (x,y,z = sum, w = count)

    DeviceBuffer<float> rayCastMinT; // Minimum t values for ray casting
    // VBO data (mapped from OpenGL)
    float* vboData = nullptr;      // Interleaved: pos(3), radius(1), color(3), quat(4), pad(3) = 14 floats    
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
    float* ballData = data.vboData + idx * 14;
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

// Compute bounds for each ball for BVH construction
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

// Compute bounds for each triangle for mesh BVH construction
__global__ void kernel_computeTriangleBounds(BallsDeviceData data) {
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
    
    float* ballData = data.vboData + idx * 14;
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
    int rootNode = data.ballsBvh.mRootNodes ? data.ballsBvh.mRootNodes[0] : -1;
    if (rootNode < 0) return;
    
    // Stack-based BVH traversal
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];
        
        // Get node bounds
        PackedNodeHalf lower = data.ballsBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = data.ballsBvh.mNodeUppers[nodeIndex];
        
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

// Ball-mesh collision using BVH
__device__ void kernel_ballMeshCollision(BallsDeviceData data, int ballIdx, int meshIdx, float searchRadius, float restitution, float dt) {

    // Get ball data (read from VBO)
    float* ballData = data.vboData + ballIdx * 14;
    Vec3 ballPos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    // Query bounds (ball position + search radius)
    Bounds3 queryBounds(ballPos - Vec3(searchRadius, searchRadius, searchRadius), 
                       ballPos + Vec3(searchRadius, searchRadius, searchRadius));
    
    // Get root node of triangle BVH
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
__global__ void kernel_ballMeshesCollision(BallsDeviceData data, float searchRadius, float restitution, float dt)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Check if we have mesh data
    if (data.numMeshTriangles == 0) return;
    
    // Get ball data (read from VBO)
    float* ballData = data.vboData + idx * 14;
    Vec3 ballPos(ballData[0], ballData[1], ballData[2]);
    
    // Query bounds (ball position + search radius)
    Bounds3 queryBounds(ballPos - Vec3(searchRadius, searchRadius, searchRadius), 
                       ballPos + Vec3(searchRadius, searchRadius, searchRadius));
    
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
                kernel_ballMeshCollision(data, idx, meshIdx, searchRadius, restitution, dt);
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
    float* ballData = data.vboData + idx * 14;
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
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Derive velocity: v = (pos - prevPos) / dt
    data.vel[idx] = (pos - data.prevPos[idx]) / dt;
}

//  Integrate quaternions
__global__ void kernel_integrateQuaternions(BallsDeviceData data, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * 14;
    
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
    
    float* ballData = data.vboData + idx * 14;
    
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


__device__ bool kernel_rayBoundsIntersection(Bounds3 bounds, Ray ray)
{
    float tEntry = -MaxFloat;
    float tExit = MaxFloat;

    for (int i = 0; i < 3; ++i)
    {
        if (ray.dir[i] != 0.0f)
        {
            float t1 = (bounds.minimum[i] - ray.orig[i]) / ray.dir[i];
            float t2 = (bounds.maximum[i] - ray.orig[i]) / ray.dir[i];

            tEntry = Max(tEntry, Min(t1, t2));
            tExit = Min(tExit, Max(t1, t2));
        }
        else if (ray.orig[i] < bounds.minimum[i] || ray.orig[i] > bounds.maximum[i])
            return false;
    }

    return tExit > 0.0f && tEntry < tExit;
}


//-----------------------------------------------------------------------------
__device__ bool kernel_rayTriangleIntersection(
    const Ray& ray, const Vec3& a, const Vec3& b, const Vec3& c, float& t, float& u, float& v)
{
    t = MaxFloat;

    Vec3 edge1, edge2, tvec, pvec, qvec;
    float det, inv_det;

    edge1 = b - a;
    edge2 = c - a;
    pvec = ray.dir.cross(edge2);
    det = edge1.dot(pvec);

    if (det == 0.0f)
        return false;
    inv_det = 1.0f / det;
    tvec = ray.orig - a;

    u = tvec.dot(pvec) * inv_det;
    if (u < 0.0f || u > 1.0f)
        return false;

    qvec = tvec.cross(edge1);
    v = ray.dir.dot(qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = edge2.dot(qvec) * inv_det;

    return true;
}


// Ball-mesh collision using BVH
__device__ bool kernel_raycastMesh(BallsDeviceData data, int meshIdx, Ray ray, float& minT) {

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
        
        if (kernel_rayBoundsIntersection(nodeBounds, ray))
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

                if (kernel_rayTriangleIntersection(ray, v0, v1, v2, t, u, v))
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


__global__ void kernel_raycast(BallsDeviceData data, bool useBalls, Ray ray)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) 
        return;

    if (useBalls) // origin is ball position
    {
        float* ballData = data.vboData + idx * 14;
        Vec3 pos(ballData[0], ballData[1], ballData[2]);
        ray.orig = pos;
    }

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

        if (kernel_rayBoundsIntersection(bounds, ray))
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

    if (useBalls)
    {
        // fade shadow depth based on hit
        float& shadowValue = data.vboData[idx * 14 + 11];
        if (minT < MaxFloat)
        {
            shadowValue = Min(1.0f, shadowValue + 0.1f);
        }
        else
        {
            shadowValue = Max(0.0f, shadowValue - 0.1f);
        }        
    }
    else
    {
        data.rayCastMinT[0] = minT;
    }
}




// Host functions callable from BallsDemo.cpp

static BVHBuilder* g_bvhBuilder = nullptr;

void BallsDemo::initCudaPhysics(GLuint vbo, cudaGraphicsResource** vboResource, BVHBuilder* bvhBuilder, Scene* scene) {
    // Ensure any previous CUDA operations are complete
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before init: %s\n", cudaGetErrorString(err));
    }
    
    deviceData->numBalls = demoDesc.numBalls;
    deviceData->worldOrig = -100.0f;
    g_bvhBuilder = bvhBuilder;
    
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
    
    // Load mesh data from scene
    if (scene && scene->getMeshCount() > 0) {
        printf("Loading mesh data for collision detection...\n");
        
        // Count total vertices and triangles
        int totalVertices = 0;
        int totalTriangles = 0;

        std::vector<int> meshFirstTriangle;
        
        for (size_t i = 0; i < scene->getMeshCount(); i++) {
            meshFirstTriangle.push_back((int)totalTriangles);
            const Mesh* mesh = scene->getMeshes()[i];
            const MeshData& data = mesh->getData();
            totalVertices += (int)(data.positions.size() / 3);
            totalTriangles += (int)(data.indices.size() / 3);
        }
        
        printf("  Total vertices: %d, Total triangles: %d\n", totalVertices, totalTriangles);
        
        if (totalTriangles > 0) {
            // Allocate host buffers
            std::vector<Vec4> meshBoundsLower(scene->getMeshCount());
            std::vector<Vec4> meshBoundsUpper(scene->getMeshCount());
            std::vector<Vec3> hostVertices(totalVertices);
            std::vector<int> hostTriIds(totalTriangles * 3);
            
            // Concatenate mesh data
            int vertexOffset = 0;
            int triangleOffset = 0;
            
            for (size_t i = 0; i < scene->getMeshCount(); i++) {
                const Mesh* mesh = scene->getMeshes()[i];
                const MeshData& data = mesh->getData();
                Bounds3 meshBounds(Empty);
                
                // Copy vertices
                int numVerts = (int)(data.positions.size() / 3);
                for (int v = 0; v < numVerts; v++) {
                    Vec3 pos(
                        data.positions[v * 3 + 0],
                        data.positions[v * 3 + 1],
                        data.positions[v * 3 + 2]
                    );
                    meshBounds.include(pos);
                    hostVertices[vertexOffset + v] = pos;
                }
                meshBoundsLower[i] = Vec4(meshBounds.minimum, 0.0f);
                meshBoundsUpper[i] = Vec4(meshBounds.maximum, 0.0f);
                
                // Copy triangle indices (offset by current vertex offset)
                int numTris = (int)(data.indices.size() / 3);
                for (int t = 0; t < numTris; t++) {
                    hostTriIds[(triangleOffset + t) * 3 + 0] = data.indices[t * 3 + 0] + vertexOffset;
                    hostTriIds[(triangleOffset + t) * 3 + 1] = data.indices[t * 3 + 1] + vertexOffset;
                    hostTriIds[(triangleOffset + t) * 3 + 2] = data.indices[t * 3 + 2] + vertexOffset;
                }
                
                vertexOffset += numVerts;
                triangleOffset += numTris;
            }
            
            // Upload to GPU using DeviceBuffer::set (combines resize + memcpy)
            deviceData->numMeshes = (int)scene->getMeshCount();
            deviceData->meshFirstTriangle.set(meshFirstTriangle);
            deviceData->meshVertices.set(hostVertices);
            deviceData->meshTriIds.set(hostTriIds);
            deviceData->numMeshTriangles = totalTriangles;

            deviceData->meshBoundsLower.set(meshBoundsLower);
            deviceData->meshBoundsUpper.set(meshBoundsUpper);
            
            // Allocate triangle bounds buffers
            deviceData->meshTriBoundsLower.resize(totalTriangles, false);
            deviceData->meshTriBoundsUpper.resize(totalTriangles, false);
            
            // Compute triangle bounds
            int numTriBlocks = (totalTriangles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            kernel_computeTriangleBounds<<<numTriBlocks, THREADS_PER_BLOCK>>>(*deviceData);
            cudaDeviceSynchronize();

            // Build BVH for meshes
            if (g_bvhBuilder) {
                printf("Building BVH for %d meshes...\n", deviceData->numMeshes);
                g_bvhBuilder->build(deviceData->meshesBvh,
                    deviceData->meshBoundsLower.buffer,
                    deviceData->meshBoundsUpper.buffer,
                    deviceData->numMeshes,
                    nullptr, 0); // No grouping
                printf("Mesh BVH built: %d nodes\n", deviceData->meshesBvh.mNumNodes);
            }
            
            // Build BVH for triangles
            if (g_bvhBuilder) {
                printf("Building BVH for %d triangles...\n", totalTriangles);
                g_bvhBuilder->build(deviceData->trianglesBvh,
                    deviceData->meshTriBoundsLower.buffer,
                    deviceData->meshTriBoundsUpper.buffer,
                    totalTriangles,
                    deviceData->meshFirstTriangle.buffer,
                    deviceData->numMeshes);
                printf("Triangle BVH built: %d nodes\n", deviceData->trianglesBvh.mNumNodes);
            }
            
            printf("Mesh collision data loaded successfully!\n");
        }
    }
    
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA initialization error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize to ensure initialization completes
    cudaDeviceSynchronize();
}

void BallsDemo::updateCudaPhysics(float dt,
                                   cudaGraphicsResource* vboResource, bool useBVH) {
    if (deviceData->numBalls == 0) return;
    
    int numBlocks = (deviceData->numBalls + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Map VBO
    float* d_vboData;
    size_t numBytes;
    cudaGraphicsMapResources(1, &vboResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vboData, &numBytes, vboResource);
    
    // Update VBO pointer
    deviceData->vboData = d_vboData;
    

    if (useBVH) {
        // BVH-based collision detection

        if (deviceData->ballBoundsLowers.size != demoDesc.numBalls) {
            deviceData->ballBoundsLowers.resize(demoDesc.numBalls, false);
            deviceData->ballBoundsUppers.resize(demoDesc.numBalls, false);
        }

        // Compute bounds for each ball
        kernel_computeBallBounds << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData);

        // Build BVH
        if (g_bvhBuilder) {
            g_bvhBuilder->build(deviceData->ballsBvh,
                deviceData->ballBoundsLowers.buffer,
                deviceData->ballBoundsUppers.buffer,
                deviceData->numBalls,
                nullptr,  // No grouping
                0);
        }
    }
    else {
        // Hash grid collision detection

        kernel_fillHash << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData);

        // Sort by hash
        thrust::device_ptr<int> hashVals(deviceData->hashVals.buffer);
        thrust::device_ptr<int> hashIds(deviceData->hashIds.buffer);
        thrust::sort_by_key(hashVals, hashVals + deviceData->numBalls, hashIds);

        deviceData->hashCellFirst.setZero();
        deviceData->hashCellLast.setZero();

        kernel_setupHash << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData);
    }

    int numSubSteps = 5;

    float sdt = dt / (float)numSubSteps;

    for (int subStep = 0; subStep < numSubSteps; subStep++)
    {

        // Run physics pipeline
        kernel_integrate << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, sdt, Vec3(0.0f, -demoDesc.gravity, 0.0f), demoDesc.friction, demoDesc.terminalVelocity);

        // Zero correction buffers before collision detection
        deviceData->posCorr.setZero();
        deviceData->newAngVel.setZero();

        if (useBVH) {
            // BVH-based collision (accumulates corrections)
            kernel_ballCollision_BVH << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, demoDesc.bounce);
        }
        else {
            // Hash grid collision detection
            kernel_ballCollision << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, demoDesc.bounce);
        }

        // Apply corrections with relaxation
        const float relaxation = 0.3f;
        kernel_applyCorrections << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, relaxation);

        kernel_wallCollision << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, demoDesc.sceneBounds, demoDesc.bounce, sdt, subStep == 0);

        // Ball-mesh collision
        if (deviceData->numMeshTriangles > 0) {
            float meshSearchRadius = 1.0f * deviceData->maxRadius;  // Search radius for mesh collisions
            kernel_ballMeshesCollision << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, meshSearchRadius, demoDesc.bounce, sdt);
        }

        // Derive velocity from position change (PBD)
        kernel_deriveVelocity << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, sdt);

        kernel_integrateQuaternions << <numBlocks, THREADS_PER_BLOCK >> > (*deviceData, sdt);
    }

    // compute ball shadow depths (writes directly to VBO at offset 11)
    Ray sunRay(Vec3(Zero), demoDesc.sunDirection);
    kernel_raycast<<<deviceData->numBalls / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*deviceData, true, sunRay);

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
    deviceData->free();
    
    // CRITICAL: Reset numBalls to 0 to clear global state
    // This prevents stray code from thinking old buffers are still valid
    deviceData->numBalls = 0;
    deviceData->numMeshTriangles = 0;
        
    // Zero out VBO pointer
    deviceData->vboData = nullptr;
    g_bvhBuilder = nullptr;
    
    // Unregister VBO
    if (vboResource) {
        cudaGraphicsUnregisterResource(vboResource);
    }
    
    // Final sync to ensure cleanup completes
    cudaDeviceSynchronize();
}

bool BallsDemo::cudaRaycast(const Ray& ray, float& minT)
{
    // Upload ray to device
    if (deviceData->numMeshTriangles == 0)
        return false;

    deviceData->rayCastMinT.resize(1, false);

    kernel_raycast<<<1 / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*deviceData, false, ray);
    deviceData->rayCastMinT.getDeviceObject(minT, 0);

    return (minT < MaxFloat);
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


