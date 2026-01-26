#include "CudaMeshes.h"
#include <cuda_runtime.h>
#include <vector>
#include "BVH.h"
#include "Geometry.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


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


__device__ bool kernel_raycastMesh(MeshesDeviceData data, int meshIdx, Ray ray, float& minT) {

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


__global__ void kernel_raycastMeshes(MeshesDeviceData data, int numRays, float* positions, Ray ray, float* hits, int stride)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numRays) 
        return;

    if (positions)
    {
        float* posPtr = positions + idx * stride;
        ray.orig = Vec3(posPtr[0], posPtr[1], posPtr[2]);
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
    hits[idx * stride] = minT;
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

            deviceData->meshBoundsLower.set(meshBoundsLower);
            deviceData->meshBoundsUpper.set(meshBoundsUpper);
            
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
                    deviceData->meshBoundsLower.buffer,
                    deviceData->meshBoundsUpper.buffer,
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

bool CudaMeshes::rayCast(int numRays, float* positions, const Ray& ray, float* hits, int stride)
{
    if (deviceData->numMeshTriangles == 0)
        return false;

    kernel_raycastMeshes << <numRays / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (*deviceData, numRays, positions, ray, hits, stride);

    return true;
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
