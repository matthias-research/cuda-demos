#include "CudaMeshes.h"
#include "CudaUtils.h"
#include <cuda_runtime.h>
#include <vector>
#include "BVH.h"
#include "Geometry.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


struct MeshesDeviceData 
{
    void free()  // no destructor because cuda would call it in the kernels
    {
        numMeshes = 0;
        numMeshTriangles = 0;

        firstTriangle.free();
        vertices.free();
        triIds.free();
        boundsLower.free();
        boundsUpper.free();
        triBoundsLower.free();
        triBoundsUpper.free();
        trianglesBvh.free();
        rayCastMinT.free();
    }
        
    int numMeshes = 0;
    int numMeshTriangles = 0;
    
    DeviceBuffer<int> firstTriangle; // Starting triangle index per mesh
    DeviceBuffer<Vec3> vertices;      // Concatenated vertices from all scene meshes
    DeviceBuffer<int> triIds;         // Triangle indices (groups of 3)
    
    DeviceBuffer<Vec4> triBoundsLower; // Lower bounds for each triangle
    DeviceBuffer<Vec4> triBoundsUpper; // Upper bounds for each triangle
    DeviceBuffer<Vec4> boundsLower;   // Overall mesh bounds
    DeviceBuffer<Vec4> boundsUpper;   // Overall mesh bounds
    BVH meshesBvh;                        // BVH structure for entire meshes
    BVH trianglesBvh;                      // Separate BVH for triangles
    DeviceBuffer<float> rayCastMinT; // Minimum t values for ray casting
};

// Compute bounds for each triangle for mesh BVH construction
__global__ void kernel_computeTriangleBounds(MeshesDeviceData data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= data.numMeshTriangles) return;
    
    // Get the three vertex indices for this triangle
    int i0 = data.triIds[idx * 3 + 0];
    int i1 = data.triIds[idx * 3 + 1];
    int i2 = data.triIds[idx * 3 + 2];
    
    // Get the three vertices
    Vec3 v0 = data.vertices[i0];
    Vec3 v1 = data.vertices[i1];
    Vec3 v2 = data.vertices[i2];
    
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
    data.triBoundsLower[idx] = Vec4(lower.x, lower.y, lower.z, 0.0f);
    data.triBoundsUpper[idx] = Vec4(upper.x, upper.y, upper.z, 0.0f);
}

__device__ void kernel_handleCollision(MeshesDeviceData data, float* pos, float* radii, int stride, float defaultRadius, int particleIdx, int meshIdx, float searchRadius) {

    float* posPtr = pos + particleIdx * stride;
    Vec3 pos(posPtr[0], posPtr[1], posPtr[2]);

    float radius = defaultRadius;
    if (radii) {
        float* radiiPtr = radii + particleIdx * stride;
        radius = radiiPtr[0];
    }
    
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
                int i0 = data.triIds[triIdx * 3 + 0];
                int i1 = data.triIds[triIdx * 3 + 1];
                int i2 = data.triIds[triIdx * 3 + 2];

                // Get the three vertices
                Vec3 v0 = data.vertices[i0];
                Vec3 v1 = data.vertices[i1];
                Vec3 v2 = data.vertices[i2];

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
__global__ void kernel_particleMeshesCollision(
    MeshesDeviceData data, int numParticles, float* particlePos, int stride, float radius, float searchRadius)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numParticles) 
    return;
    
    if (data.numMeshTriangles == 0) return;
    
    float* particleData = particlePos + idx * stride;
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
                kernel_particleMeshCollision(data, particlePos, stride, radius, idx, meshIdx, searchRadius);
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

__device__ bool kernel_raycast(MeshesDeviceData data, int meshIdx, Ray ray, float& minT) {

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
                int i0 = data.triIds[triIdx * 3 + 0];
                int i1 = data.triIds[triIdx * 3 + 1];
                int i2 = data.triIds[triIdx * 3 + 2];

                // Get the three vertices
                Vec3 v0 = data.vertices[i0];
                Vec3 v1 = data.vertices[i1];
                Vec3 v2 = data.vertices[i2];

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


__global__ void kernel_raycastMeshes(MeshesDeviceData data, Ray ray)
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

// Host functions 

void CudaMeshes::initialize(const Scene* scene) 
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before init: %s\n", cudaGetErrorString(err));
    }

    if (!deviceData) 
        deviceData = std::make_shared<MeshesDeviceData>();
    if (!bvhBuilder) 
        bvhBuilder = std::make_shared<BVHBuilder>();

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
            deviceData->firstTriangle.set(meshFirstTriangle);
            deviceData->vertices.set(hostVertices);
            deviceData->triIds.set(hostTriIds);
            deviceData->numMeshTriangles = totalTriangles;

            deviceData->boundsLower.set(meshBoundsLower);
            deviceData->boundsUpper.set(meshBoundsUpper);
            
            // Allocate triangle bounds buffers
            deviceData->triBoundsLower.resize(totalTriangles, false);
            deviceData->triBoundsUpper.resize(totalTriangles, false);
            
            // Compute triangle bounds
            int numTriBlocks = (totalTriangles + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            kernel_computeTriangleBounds<<<numTriBlocks, THREADS_PER_BLOCK>>>(*deviceData);
            cudaDeviceSynchronize();

            // Build BVH for meshes
            if (bvhBuilder) {
                printf("Building BVH for %d meshes...\n", deviceData->numMeshes);
                bvhBuilder->build(deviceData->meshesBvh,
                    deviceData->boundsLower.buffer,
                    deviceData->boundsUpper.buffer,
                    deviceData->numMeshes,
                    nullptr, 0); // No grouping
                printf("Mesh BVH built: %d nodes\n", deviceData->meshesBvh.mNumNodes);
            }
            
            // Build BVH for triangles
            if (bvhBuilder) {
                printf("Building BVH for %d triangles...\n", totalTriangles);
                bvhBuilder->build(deviceData->trianglesBvh,
                    deviceData->triBoundsLower.buffer,
                    deviceData->triBoundsUpper.buffer,
                    totalTriangles,
                    deviceData->firstTriangle.buffer,
                    deviceData->numMeshes);
                printf("Triangle BVH built: %d nodes\n", deviceData->trianglesBvh.mNumNodes);
            }
            
            printf("Mesh collision data loaded successfully!\n");
        }
    }
}

bool CudaMeshes::rayCast(const Ray& ray, float& minT)
{
    if (deviceData->numMeshTriangles == 0)
        return false;

    deviceData->rayCastMinT.resize(1, false);

    kernel_raycast << <1 / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (*deviceData, false, ray);
    deviceData->rayCastMinT.getDeviceObject(minT, 0);

    return (minT < MaxFloat);
}

void CudaMeshes::handleCollisions(int numParticles, float* positions, int stride,
    float* radii, float defaultRadius, float searchDist)
{
    kernel_meshesCollision << <m_deviceData->numPoints / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (
        *m_deviceData);


}


void CudaMeshes::cleanup() 
{
    if (deviceData) {
        deviceData->free();
        deviceData.reset();
    }
    if (bvhBuilder) {
        bvhBuilder.reset();
    }
}
