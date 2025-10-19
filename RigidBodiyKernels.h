#include "Vec.h"
#include "Geometry.h"
#include "CudaUtils.h"
#include "BVH.h"

struct RigidBodyParams
{
    float timeStep = 0.016f;
    float contactBand = 0.01f;
};
struct RigidBody
{
    Transform transform;
    Transform prevTransform;

    Vec3 velocity;
    Vec3 angularVelocity;

    float invMass;
    Vec3 invInertia;

    int firstFace;
    int numFaces;

    int firstVertex;
    int numVertices;

    Bounds3 localBounds;
};

struct RigidBodyContact
{
    int body0;
    int body1;
    Vec3 pos0;
    Vec3 pos1;
    Vec3 normal;
};

struct RigidBodyDeviceData
{
    RigidBodyDeviceData()
    {
        clear();
    }

    void clear()
    {
        numBodies = 0;
        numContacts = 0;
    }

    void free()
    {
        bodies.free();
        vertices.free();
        faceFirstIndex.free();
        faceIndices.free();
        facePlanes.free();
        faceCenters.free();
        faceNeighbors.free();
        contacts.free();
        bodyLowers.free();
        bodyUppers.free();
        bvh.free();
        clear();
    }

    size_t getAllocationSizes() const
    {
        size_t s = 0;
        s += bodies.capacity * sizeof(RigidBody);
        s += vertices.capacity * sizeof(Vec3);
        s += faceFirstIndex.capacity * sizeof(int);
        s += faceIndices.capacity * sizeof(int);
        s += facePlanes.capacity * sizeof(Plane);
        s += faceCenters.capacity * sizeof(Vec3);
        s += faceNeighbors.capacity * sizeof(int);
        s += contacts.capacity * sizeof(RigidBodyContact);
        s += bodyLowers.capacity * sizeof(Vec4);
        s += bodyUppers.capacity * sizeof(Vec4);
        s += bvh.getAllocationSize();
        return s;
    }

    RigidBodyParams params;

    int numBodies;
    int numContacts;

    // body data
    DeviceBuffer<RigidBody> bodies;
    DeviceBuffer<Vec3> vertices;
    DeviceBuffer<Plane> facePlanes;
    DeviceBuffer<Vec3> faceCenters;
    DeviceBuffer<int> faceFirstIndex;
    DeviceBuffer<int> faceIndices;
    DeviceBuffer<int> faceNeighbors;

    // contact data
    DeviceBuffer<RigidBodyContact> contacts;

    // acceleration structures
    BVH bvh;
    DeviceBuffer<Vec4> bodyLowers;
    DeviceBuffer<Vec4> bodyUppers;
};

__global__ void device_computeBodyWorldBounds(RigidBodyDeviceData data)
{
    int bodyNr = threadIdx.x + blockIdx.x * blockDim.x;

    if (bodyNr >= data.numBodies)
        return;

    RigidBody body = data.bodies[bodyNr];
    Bounds3 worldBounds = body.localBounds.transform(body.transform);

    data.bodyLowers[bodyNr] = Vec4(worldBounds.minimum);
    data.bodyUppers[bodyNr] = Vec4(worldBounds.maximum);
}


__device__ void device_findFaceContactPlane(RigidBodyDeviceData data, int bodyNr0, int bodyNr1, float& minDepth, Plane& minPlane0, Plane& minPlane1)
{
    RigidBody body0 = data.bodies[bodyNr0];
    RigidBody body1 = data.bodies[bodyNr1];

    for (int i = 0; i < body0.numFaces; i++)
    {
        Plane facePlane = data.facePlanes[body0.firstFace + i];
        // working in the frame of body1
        Plane worldPlane = body0.transform(facePlane);
        Plane plane = body1.transformInv(worldPlane);

        float maxDepth = -MaxFloat;

        for (int j = 0; j < body1.numVertices; j++)
        {
            Vec3 p = data.vertices[body1.firstVertex + j];
            float depth = -plane.height(p);
            maxDepth = max(maxDepth, depth);

            if (maxDepth > minDepth)
                break;
        }
        if (maxDepth < minDepth)
        {
            minDepth = maxDepth;
            minPlane0 = worldPlane;
            minPlane1 = worldPlane;
            minPlane1.d -= maxDepth;

            if (minDepth < -data.params.contactBand) // separation axis found
                return;
        }
    }
}

__device__ void device_findEdgeContactPlane(RigidBodyDeviceData data, int bodyNr0, int bodyNr1, float& minDepth, Plane& minPlane0, Plane& minPlane1)
{
    const float eps = 1e-5f;

    RigidBody body0 = data.bodies[bodyNr0];
    RigidBody body1 = data.bodies[bodyNr1];

    for (int faceNr0 = 0; faceNr0 < body0.numFaces; faceNr0++)
    {
        int start0 = data.faceFirstIndex[body0.firstFace + faceNr0];
        int size0 = data.faceFirstIndex[body0.firstFace + faceNr0 + 1] - start0;

        for (int i = 0; i < size0; i++)
        {
            int neighbor0 = data.faceNeighbors[start0 + i];
            if (neighbor0 < faceNr0)
                continue; // already processed

            int id0 = data.faceIndices[start0 + i];
            int id1 = data.faceIndices[start0 + (i + 1) % size0];

            Vec3 p0 = body0.transform(data.vertices[body0.firstVertex + id0]);
            Vec3 p1 = body0.transform(data.vertices[body0.firstVertex + id1]);

            Vec3 cp0 = body0.transform(data.faceCenters[body0.firstFace + faceNr0]);
            Vec3 cp1 = body0.transform(data.faceCenters[body0.firstFace + neighbor0]);

            for (int faceNr1 = 0; faceNr1 < body1.numFaces; faceNr1++)
            {
                int start1 = data.faceFirstIndex[body1.firstFace + faceNr1];
                int size1 = data.faceFirstIndex[body1.firstFace + faceNr1 + 1] - start1;

                for (int j = 0; j < size1; j++)
                {
                    int neighbor1 = data.faceNeighbors[start1 + j];
                    if (neighbor1 < faceNr1)
                        continue; // already processed

                    int jd0 = data.faceIndices[start1 + j];
                    int jd1 = data.faceIndices[start1 + (j + 1) % size1];

                    Vec3 q0 = body1.transform(data.vertices[body1.firstVertex + jd0]);
                    Vec3 q1 = body1.transform(data.vertices[body1.firstVertex + jd1]);

                    Vec3 cq0 = body1.transform(data.faceCenters[body1.firstFace + faceNr1]);
                    Vec3 cq1 = body1.transform(data.faceCenters[body1.firstFace + neighbor1]);

                    Vec3 n = (p1 - p0).cross(q1 - q0).normalized();

                    if (n.dot(p0 - cp0) < 0.0f) 
                        n = -n;
                        
                    float depth = n.dot(p0 - q0);

                    if (depth >= minDepth)
                        continue;

                    if (n.dot(p1 - cp1) < -eps)
                        continue;

                    if (n.dot(q0 - cq0) > eps)
                        continue;

                    if (n.dot(q1 - cq1) < -eps)
                        continue;

                    // float pt, qt;

                    // if (!getClosestPointsOnRays(Ray(p0, p1 - p0), Ray(q0, q1 - q0), pt, qt))
                    //     continue;

                    // if (pt < -eps || pt > 1.0f + eps || qt < -eps || qt > 1.0f + eps)
                    //     continue;

                    if (depth < minDepth)
                    {
                        minDepth = depth;
                        minPlane0 = Plane(n, n.dot(p0));
                        minPlane1 = Plane(n, n.dot(q0));
                    }
                }
            }
        }
    }
}

__device__ void device_createPairContactPoints(RigidBodyDeviceData data, int bodyNr0, int bodyNr1)
{
    float minDepth = MaxFloat;
    Plane minPlane0;
    Plane minPlane1;

    device_findFaceContactPlane(data, bodyNr0, bodyNr1, minDepth, minPlane0, minPlane1);
    device_findFaceContactPlane(data, bodyNr1, bodyNr0, minDepth, minPlane1, minPlane0);

    device_findEdgeContactPlane(data, bodyNr0, bodyNr1, minDepth, minPlane0, minPlane1);
}

__global__ void device_createContactPoints(RigidBodyDeviceData data)
{
    int bodyNr0 = threadIdx.x + blockIdx.x * blockDim.x;

    if (bodyNr0 >= data.numBodies)
        return;

    Bounds3 bodyBounds = Bounds3(Vec3(data.bodyLowers[bodyNr0]), Vec3(data.bodyUppers[bodyNr0]));

    int stack[32];
    stack[0] = data.bvh.mRootNodes[0];
    int count = 1;

    while (count)
    {
        const int nodeIndex = stack[--count];

        PackedNodeHalf lower = data.bvh.mNodeLowers[nodeIndex];
        PackedNodeHalf upper = data.bvh.mNodeUppers[nodeIndex];

        Bounds3 bounds(Vec3(lower.x, lower.y, lower.z), Vec3(upper.x, upper.y, upper.z));

        if (bounds.intersect(bodyBounds))
        {
            const int leftIndex = lower.i;
            const int rightIndex = upper.i;

            if (lower.b)
            {                
                int bodyNr1 = leftIndex;
                if (bodyNr1 > bodyNr0)
                {
                    device_createPairContactPoints(data, bodyNr0, bodyNr1);
                }
            }
            else
            {
                stack[count++] = leftIndex;
                stack[count++] = rightIndex;
            }
        }
    }
}
