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
static const int HASH_SIZE = 370111;  // Prime number for better distribution

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
        angVelCorr.free();
        meshVertices.free();
        meshTriIds.free();
        meshTriBoundsLower.free();
        meshTriBoundsUpper.free();
        numBalls = 0;
        numMeshTriangles = 0;
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
        s += angVelCorr.allocationSize();
        s += meshVertices.allocationSize();
        s += meshTriIds.allocationSize();
        s += meshTriBoundsLower.allocationSize();
        s += meshTriBoundsUpper.allocationSize();
        return s;
    }
    
    int numBalls = 0;
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
    BVH bvh;                              // BVH structure
    
    // Mesh collision data
    DeviceBuffer<Vec3> meshVertices;      // Concatenated vertices from all scene meshes
    DeviceBuffer<int> meshTriIds;         // Triangle indices (groups of 3)
    int numMeshTriangles = 0;             // Total number of triangles
    
    // Triangle BVH
    DeviceBuffer<Vec4> meshTriBoundsLower; // Lower bounds for each triangle
    DeviceBuffer<Vec4> meshTriBoundsUpper; // Upper bounds for each triangle
    BVH meshBvh;                           // BVH structure for mesh triangles
    
    // Collision correction buffers (for symmetric resolution)
    DeviceBuffer<Vec3> posCorr;       // Position corrections
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

// Kernel 1: Integrate - save previous position and predict new position (PBD)
__global__ void kernel_integrate(BallsDeviceData data, float dt, Vec3 gravity, float friction) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read from VBO
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Store previous position
    data.prevPos[idx] = pos;
    
    // Apply friction to velocity
    data.vel[idx] *= friction;
    data.angVel[idx] *= friction;
    
    // Predict position: pos = pos + vel * dt + gravity * dt^2
    pos += data.vel[idx] * dt + gravity * (dt * dt);
    
    // Write predicted position back to VBO
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

// Kernel: Compute bounds for each triangle for mesh BVH construction
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

// Kernel 5: Wall collision (generalized for arbitrary planes)
__global__ void kernel_wallCollision(BallsDeviceData data, float roomSize, float bounce) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    float halfRoom = roomSize * 0.5f;
    
    // Define 5 walls as planes (normal points inward, n·p + d = 0 on plane)
    // Plane equation: n·p + d = 0, where positive values of (n·p + d) mean "inside the room"
    Plane walls[5] = {
        Plane(Vec3(-1, 0, 0),  halfRoom),    // +X wall at x = +halfRoom (normal points left)
        Plane(Vec3(1, 0, 0),   halfRoom),    // -X wall at x = -halfRoom (normal points right)
        Plane(Vec3(0, 1, 0),   0),           // Floor at y = 0 (normal points up)
//        Plane(Vec3(0, -1, 0),  roomSize),    // Ceiling at y = roomSize (normal points down)
        Plane(Vec3(0, 0, -1),  halfRoom),    // +Z wall at z = +halfRoom (normal points forward)
        Plane(Vec3(0, 0, 1),   halfRoom)     // -Z wall at z = -halfRoom (normal points back)
    };
    
    // Check collision with each wall
    for (int i = 0; i < 5; i++) {
        const Plane& wall = walls[i];
        
        // Signed distance from ball center to plane (positive = inside room, negative = outside)
        float signedDist = wall.n.dot(pos) + wall.d;
        float penetration = radius - signedDist;
        
        if (penetration > 0) {
            // Position correction: push ball back inside the room (in direction of inward normal)
            pos += wall.n * penetration;
        }
    }
    
    // Write back corrected position
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

// Kernel: Ball-mesh collision using BVH
__global__ void kernel_ballMeshCollision(BallsDeviceData data, float searchRadius) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Check if we have mesh data
    if (data.numMeshTriangles == 0) return;
    
    // Get ball data (read from VBO)
    float* ballData = data.vboData + idx * 14;
    Vec3 ballPos(ballData[0], ballData[1], ballData[2]);
    float radius = ballData[3];
    
    // Query bounds (ball position + search radius)
    Bounds3 queryBounds(ballPos - Vec3(searchRadius, searchRadius, searchRadius), 
                       ballPos + Vec3(searchRadius, searchRadius, searchRadius));
    
    // Get root node of mesh BVH
    int rootNode = data.meshBvh.mRootNodes ? data.meshBvh.mRootNodes[0] : -1;
    if (rootNode < 0) return;
    
    // Track closest point and triangle
    float minDistSq = searchRadius * searchRadius;
    int closestTriIdx = -1;
    Vec3 closestPoint;
    
    // Stack-based BVH traversal
    int stack[64];
    stack[0] = rootNode;
    int stackCount = 1;
    
    while (stackCount > 0) {
        int nodeIndex = stack[--stackCount];
        
        // Get node bounds
        PackedNodeHalf lower = data.meshBvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = data.meshBvh.mNodeUppers[nodeIndex];
        
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
                
                // Check if this point is closer than previous closest
                float distSq = (point - ballPos).magnitudeSquared();
                if (distSq < minDistSq) {
                    minDistSq = distSq;
                    closestTriIdx = triIdx;
                    closestPoint = point;
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
    
    // Handle collision if we found a closest triangle
    if (closestTriIdx >= 0) {
        // Compute distance and direction from closest point to ball center
        Vec3 delta = ballPos - closestPoint;
        float dist = delta.magnitude();
        
        if (dist > 0.001f) {  // Avoid division by zero
            Vec3 direction = delta / dist;  // Normalized vector pointing from surface to ball center
            
            // Get triangle normal to determine if ball is inside or outside
            int i0 = data.meshTriIds[closestTriIdx * 3 + 0];
            int i1 = data.meshTriIds[closestTriIdx * 3 + 1];
            int i2 = data.meshTriIds[closestTriIdx * 3 + 2];
            
            Vec3 v0 = data.meshVertices[i0];
            Vec3 v1 = data.meshVertices[i1];
            Vec3 v2 = data.meshVertices[i2];
            
            // Compute triangle normal
            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            Vec3 triNormal = edge1.cross(edge2);
            triNormal = triNormal / triNormal.magnitude();  // Normalize
            
            // Determine if ball is inside or outside mesh
            // If direction and normal point in similar direction (dot > 0), ball is outside
            float alignment = direction.dot(triNormal);
            
            if (alignment > 0.0f) {
                // Ball is outside the mesh
                if (dist < radius) {
                    // Collision: push ball out along direction
                    float penetration = radius - dist;
                    ballPos += direction * penetration;
                }
            } else {
                // Ball is inside the mesh - push it out
                float penetration = dist + radius;
                ballPos += -direction * penetration;  // Go against direction to exit
            }
            
            // Write corrected position back to VBO
            ballData[0] = ballPos.x;
            ballData[1] = ballPos.y;
            ballData[2] = ballPos.z;
        }
    }
}

// Kernel: Apply collision corrections with relaxation (PBD)
__global__ void kernel_applyCorrections(BallsDeviceData data, float relaxation) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read from VBO
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Apply position corrections with relaxation factor
    pos += data.posCorr[idx] * relaxation;
    data.angVel[idx] += data.angVelCorr[idx] * relaxation;
    
    // Write position back to VBO
    ballData[0] = pos.x;
    ballData[1] = pos.y;
    ballData[2] = pos.z;
}

// Kernel: Derive velocity from position change (PBD)
__global__ void kernel_deriveVelocity(BallsDeviceData data, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numBalls) return;
    
    // Read current position from VBO
    float* ballData = data.vboData + idx * 14;
    Vec3 pos(ballData[0], ballData[1], ballData[2]);
    
    // Derive velocity: v = (pos - prevPos) / dt
    data.vel[idx] = (pos - data.prevPos[idx]) / dt;
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
//    quat = quat.rotateLinear(quat, omega);
    
    // Write back
    ballData[7] = quat.w;
    ballData[8] = quat.x;
    ballData[9] = quat.y;
    ballData[10] = quat.z;
}

// Initialization kernel - sets up balls on a grid
__global__ void kernel_initBalls(BallsDeviceData data, float roomSize, float minRadius, float maxRadius, 
                                  int ballsPerLayer, int ballsPerRow, float gridSpacing, unsigned long seed) {
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
    
    // Grid-based position
    int layer = idx / ballsPerLayer;
    int inLayer = idx % ballsPerLayer;
    int row = inLayer / ballsPerRow;
    int col = inLayer % ballsPerRow;
    
    float halfRoom = roomSize * 0.5f;
    ballData[0] = -halfRoom + gridSpacing + col * gridSpacing;  // x
    ballData[2] = -halfRoom + gridSpacing + row * gridSpacing;  // z
    ballData[1] = maxRadius + layer * gridSpacing;              // y (start from floor + radius)
    
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

extern "C" void initCudaPhysics(int numBalls, float roomSize, float minRadius, float maxRadius, GLuint vbo, cudaGraphicsResource** vboResource, BVHBuilder* bvhBuilder, Scene* scene) {
    g_deviceData.numBalls = numBalls;
    g_deviceData.worldOrig = -100.0f;
    g_bvhBuilder = bvhBuilder;
    
    // Use provided maximum radius
    g_deviceData.maxRadius = maxRadius;
    g_deviceData.gridSpacing = 2.5f * maxRadius;
    
    // Allocate physics arrays using DeviceBuffer
    g_deviceData.vel.resize(numBalls, false);
    g_deviceData.prevPos.resize(numBalls, false);
    g_deviceData.angVel.resize(numBalls, false);
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
    g_deviceData.angVelCorr.resize(numBalls, false);
    
    // Initialize memory
    g_deviceData.vel.setZero();
    g_deviceData.angVel.setZero();
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
    
    // Calculate grid layout parameters
    float gridSpacing = 2.0f * maxRadius;  // Space between ball centers
    float halfRoom = roomSize * 0.5f;
    float usableWidth = roomSize - 2.0f * gridSpacing;  // Leave margin from walls
    int ballsPerRow = (int)(usableWidth / gridSpacing);
    if (ballsPerRow < 1) ballsPerRow = 1;
    int ballsPerLayer = ballsPerRow * ballsPerRow;
    
    // Initialize balls with grid data
    int numBlocks = (numBalls + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching kernel_initBalls with %d blocks, %d threads per block, %d total balls\n", numBlocks, THREADS_PER_BLOCK, numBalls);
    printf("Grid layout: %d balls per row, %d balls per layer, spacing: %.2f\n", ballsPerRow, ballsPerLayer, gridSpacing);
    
    kernel_initBalls<<<numBlocks, THREADS_PER_BLOCK>>>(g_deviceData, roomSize, minRadius, maxRadius, 
                                                        ballsPerLayer, ballsPerRow, gridSpacing, time(nullptr));
    
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
        
        for (size_t i = 0; i < scene->getMeshCount(); i++) {
            const Mesh* mesh = scene->getMeshes()[i];
            const MeshData& data = mesh->getData();
            totalVertices += data.positions.size() / 3;
            totalTriangles += data.indices.size() / 3;
        }
        
        printf("  Total vertices: %d, Total triangles: %d\n", totalVertices, totalTriangles);
        
        if (totalTriangles > 0) {
            // Allocate host buffers
            std::vector<Vec3> hostVertices(totalVertices);
            std::vector<int> hostTriIds(totalTriangles * 3);
            
            // Concatenate mesh data
            int vertexOffset = 0;
            int triangleOffset = 0;
            
            for (size_t i = 0; i < scene->getMeshCount(); i++) {
                const Mesh* mesh = scene->getMeshes()[i];
                const MeshData& data = mesh->getData();
                
                // Copy vertices
                int numVerts = data.positions.size() / 3;
                for (int v = 0; v < numVerts; v++) {
                    hostVertices[vertexOffset + v] = Vec3(
                        data.positions[v * 3 + 0],
                        data.positions[v * 3 + 1],
                        data.positions[v * 3 + 2]
                    );
                }
                
                // Copy triangle indices (offset by current vertex offset)
                int numTris = data.indices.size() / 3;
                for (int t = 0; t < numTris; t++) {
                    hostTriIds[(triangleOffset + t) * 3 + 0] = data.indices[t * 3 + 0] + vertexOffset;
                    hostTriIds[(triangleOffset + t) * 3 + 1] = data.indices[t * 3 + 1] + vertexOffset;
                    hostTriIds[(triangleOffset + t) * 3 + 2] = data.indices[t * 3 + 2] + vertexOffset;
                }
                
                vertexOffset += numVerts;
                triangleOffset += numTris;
            }
            
            // Upload to GPU
            g_deviceData.meshVertices.resize(totalVertices, false);
            g_deviceData.meshTriIds.resize(totalTriangles * 3, false);
            g_deviceData.numMeshTriangles = totalTriangles;
            
            cudaMemcpy(g_deviceData.meshVertices.buffer, hostVertices.data(), 
                      totalVertices * sizeof(Vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(g_deviceData.meshTriIds.buffer, hostTriIds.data(), 
                      totalTriangles * 3 * sizeof(int), cudaMemcpyHostToDevice);
            
            // Allocate triangle bounds buffers
            g_deviceData.meshTriBoundsLower.resize(totalTriangles, false);
            g_deviceData.meshTriBoundsUpper.resize(totalTriangles, false);
            
            // Compute triangle bounds
            int numTriBlocks = (totalTriangles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            kernel_computeTriangleBounds<<<numTriBlocks, THREADS_PER_BLOCK>>>(g_deviceData);
            cudaDeviceSynchronize();
            
            // Build BVH for triangles
            if (g_bvhBuilder) {
                printf("Building BVH for %d triangles...\n", totalTriangles);
                g_bvhBuilder->build(g_deviceData.meshBvh,
                    g_deviceData.meshTriBoundsLower.buffer,
                    g_deviceData.meshTriBoundsUpper.buffer,
                    totalTriangles,
                    nullptr,  // No grouping
                    0);
                printf("Mesh BVH built: %d nodes\n", g_deviceData.meshBvh.mNumNodes);
            }
            
            printf("Mesh collision data loaded successfully!\n");
        }
    }
    
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
    

    if (useBVH) {
        // BVH-based collision detection

        // Compute bounds for each ball
        kernel_computeBallBounds << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData);

        // Build BVH
        if (g_bvhBuilder) {
            g_bvhBuilder->build(g_deviceData.bvh,
                g_deviceData.ballBoundsLowers.buffer,
                g_deviceData.ballBoundsUppers.buffer,
                g_deviceData.numBalls,
                nullptr,  // No grouping
                0);
        }
    }
    else {
        // Hash grid collision detection

        kernel_fillHash << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData);

        // Sort by hash
        thrust::device_ptr<int> hashVals(g_deviceData.hashVals.buffer);
        thrust::device_ptr<int> hashIds(g_deviceData.hashIds.buffer);
        thrust::sort_by_key(hashVals, hashVals + g_deviceData.numBalls, hashIds);

        g_deviceData.hashCellFirst.setZero();
        g_deviceData.hashCellLast.setZero();

        kernel_setupHash << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData);
    }

    int numSubSteps = 5;

    float sdt = dt / (float)numSubSteps;

    for (int subStep = 0; subStep < numSubSteps; subStep++)
    {

        // Run physics pipeline
        kernel_integrate << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData, sdt, gravity, friction);

        // Zero correction buffers before collision detection
        g_deviceData.posCorr.setZero();
        g_deviceData.angVelCorr.setZero();

        if (useBVH) {
            // BVH-based collision (accumulates corrections)
            kernel_ballCollision_BVH << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData, bounce);
        }
        else {
            // Hash grid collision detection
            kernel_ballCollision << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData, bounce);
        }

        // Apply corrections with relaxation
        const float relaxation = 0.3f;
        kernel_applyCorrections << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData, relaxation);

        kernel_wallCollision << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData, roomSize, bounce);

        // Ball-mesh collision
        if (g_deviceData.numMeshTriangles > 0) {
            float meshSearchRadius = 2.0f * g_deviceData.maxRadius;  // Search radius for mesh collisions
            kernel_ballMeshCollision << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData, meshSearchRadius);
        }

        // Derive velocity from position change (PBD)
        kernel_deriveVelocity << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData, sdt);

        kernel_integrateQuaternions << <numBlocks, THREADS_PER_BLOCK >> > (g_deviceData, sdt);
    }

    
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
    
    // Free BVHs
    g_deviceData.bvh.free();
    g_deviceData.meshBvh.free();
    
    // Zero out VBO pointer
    g_deviceData.vboData = nullptr;
    g_bvhBuilder = nullptr;
    
    // Unregister VBO
    if (vboResource) {
        cudaGraphicsUnregisterResource(vboResource);
    }
}

