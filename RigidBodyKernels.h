#include "Vec.h"
#include "Geometry.h"
#include "CudaUtils.h"
// Temporarily disabled: #include "BVH.h"

struct RigidBodyParams
{
    float timeStep = 0.016f;
    float contactBand = 0.01f;
};
struct RigidBody
{
    Transform pose;
    Transform prevPose;

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
        // Temporarily disabled: bvh.free();
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
        // s += bvh.getAllocationSize();
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
    // Temporarily disabled: BVH bvh;
    DeviceBuffer<Vec4> bodyLowers;
    DeviceBuffer<Vec4> bodyUppers;
};

__global__ void device_computeBodyWorldBounds(RigidBodyDeviceData data)
{
    int bodyNr = threadIdx.x + blockIdx.x * blockDim.x;

    if (bodyNr >= data.numBodies)
        return;

    RigidBody body = data.bodies[bodyNr];
    Bounds3 worldBounds = body.localBounds.transform(body.pose);

    data.bodyLowers[bodyNr] = Vec4(worldBounds.minimum);
    data.bodyUppers[bodyNr] = Vec4(worldBounds.maximum);
}


__device__ void device_findFaceContactPlane(RigidBodyDeviceData data, int bodyNr0, int bodyNr1, 
    float& minDepth, int& minFaceNr0, int& minFaceNr1, Vec3& normal, bool swap = false)
{
    RigidBody body0 = data.bodies[bodyNr0];
    RigidBody body1 = data.bodies[bodyNr1];

    for (int i = 0; i < body0.numFaces; i++)
    {
        Plane facePlane = data.facePlanes[body0.firstFace + i];
        // working in the frame of body1
        Plane worldPlane = body0.pose.transform(facePlane);
        Plane plane = body1.pose.transformInv(worldPlane);

        float maxDepth = -MaxFloat;

        for (int j = 0; j < body1.numVertices; j++)
        {
            Vec3 p = data.vertices[body1.firstVertex + j];
            float depth = -plane.height(p);
            maxDepth = Max(maxDepth, depth);

            if (maxDepth > minDepth)
                break;
        }
        if (maxDepth < minDepth)
        {
            minDepth = maxDepth;
            minFaceNr0 = i;
            minFaceNr1 = -1; // will determine if this plane wins
            bool swapped = false;    // todo
            normal = swapped ? -worldPlane.n : worldPlane.n;

            if (minDepth < -data.params.contactBand) // separation axis found
                return;
        }
    }
}

__device__ void device_findEdgeContactPlane(RigidBodyDeviceData data, int bodyNr0, int bodyNr1, 
    float& minDepth, int& minFaceNr0, int& minFaceNr1, Vec3& normal)
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

            int id00 = data.faceIndices[start0 + i];
            int id01 = data.faceIndices[start0 + (i + 1) % size0];

            Vec3 p00 = body0.pose.transform(data.vertices[body0.firstVertex + id00]);
            Vec3 p01 = body0.pose.transform(data.vertices[body0.firstVertex + id01]);

            Vec3 c00 = body0.pose.transform(data.faceCenters[body0.firstFace + faceNr0]);
            Vec3 c01 = body0.pose.transform(data.faceCenters[body0.firstFace + neighbor0]);

            Vec3 fn00 = body0.pose.q.rotate(data.facePlanes[body0.firstFace + faceNr0].n);
            Vec3 fn01 = body0.pose.q.rotate(data.facePlanes[body0.firstFace + neighbor0].n);

            for (int faceNr1 = 0; faceNr1 < body1.numFaces; faceNr1++)
            {
                int start1 = data.faceFirstIndex[body1.firstFace + faceNr1];
                int size1 = data.faceFirstIndex[body1.firstFace + faceNr1 + 1] - start1;

                for (int j = 0; j < size1; j++)
                {
                    int neighbor1 = data.faceNeighbors[start1 + j];
                    if (neighbor1 < faceNr1)
                        continue; // already processed

                    int id10 = data.faceIndices[start1 + j];
                    int id11 = data.faceIndices[start1 + (j + 1) % size1];

                    Vec3 p10 = body1.pose.transform(data.vertices[body1.firstVertex + id10]);
                    Vec3 p11 = body1.pose.transform(data.vertices[body1.firstVertex + id11]);

                    Vec3 c10 = body1.pose.transform(data.faceCenters[body1.firstFace + faceNr1]);
                    Vec3 c11 = body1.pose.transform(data.faceCenters[body1.firstFace + neighbor1]);

                    Vec3 n = (p01 - p00).cross(p11 - p10).normalized();

                    if (n.dot(p00 - c00) < 0.0f) 
                        n = -n;

                    float depth = n.dot(p00 - p10);

                    if (depth >= minDepth)
                        continue;

                    if (n.dot(p00 - c01) < -eps)
                        continue;

                    if (n.dot(p10 - c10) > eps)
                        continue;

                    if (n.dot(p10 - c11) < -eps)
                        continue;

                    // float pt, qt;

                    // if (!getClosestPointsOnRays(Ray(p0, p1 - p0), Ray(q0, q1 - q0), pt, qt))
                    //     continue;

                    // if (pt < -eps || pt > 1.0f + eps || qt < -eps || qt > 1.0f + eps)
                    //     continue;

                    if (depth < minDepth)
                    {
                        normal = n;

                        Vec3 fn10 = body1.pose.q.rotate(data.facePlanes[body1.firstFace + faceNr1].n);
                        Vec3 fn11 = body1.pose.q.rotate(data.facePlanes[body1.firstFace + neighbor1].n);

                        minFaceNr0 = n.dot(fn00) < n.dot(fn01) ? faceNr0 : neighbor0;
                        minFaceNr1 = n.dot(fn10) < n.dot(fn11) ? faceNr1 : neighbor1;
                    }
                }
            }
        }
    }
}

__device__ void device_createPairContactPoints(RigidBodyDeviceData data, int bodyNr0, int bodyNr1)
{
    float minDepth = MaxFloat;
    Vec3 normal;
    int minFaceNr0 = -1;
    int minFaceNr1 = -1;

    //device_findFaceContactPlane(data, bodyNr0, bodyNr1, minDepth, minFaceNr0, minFaceNr1, normal);
    //device_findFaceContactPlane(data, bodyNr1, bodyNr0, minDepth, minFaceNr1, minFaceNr0, normal, true);
    //device_findEdgeContactPlane(data, bodyNr0, bodyNr1, minDepth, minFaceNr0, minFaceNr1, normal);

    //if (minFaceNr1 < 0)
    //{
    //    
    //}
}

__global__ void device_createContactPoints(RigidBodyDeviceData data)
{
    // Temporarily disabled - requires BVH
    /*
    int bodyNr0 = threadIdx.x + blockIdx.x * blockDim.x;

    if (bodyNr0 >= data.numBodies)
        return;

    Bounds3 bodyBounds = Bounds3(data.bodyLowers[bodyNr0].getXYZ(), data.bodyUppers[bodyNr0].getXYZ());

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
    */
}
