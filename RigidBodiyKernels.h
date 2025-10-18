#include "Vec.h"
#include "CudaUtils.h"
#include "BVH.h"

struct RigidBody
{
    Transform transform;
    Transform prevTransform;

    Vec3 velocity;
    Vec3 angularVelocity;

    Vec3 extent;
    float invMass;
    Vec3 invInertia;
}

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
        contacts.free();
        bodyLowers.free();
        bodyUppers.free();
        bvh.free();
        clear();
    }

    void getAllocationSizes() const
    {
        size_t s = 0;
        s += bodies.getAllocationSize();
        s += contacts.getAllocationSize();
        s += bodyLowers.getAllocationSize();
        s += bodyUppers.getAllocationSize();
        s += bvh.size();
        return s;
    }

    int numBodies;
    int numContacts;

    DeviceBuffer<RigidBody> bodies;
    DeviceBuffer<RigidBodyContact> contacts;
    DeviceBuffer<Vec4> bodyLowers;
    DeviceBuffer<Vec4> bodyUppers;
    BVH bvh;
};

__global__ void device_computeBodyBounds(RigidBodyDeviceData data)
{
    int bodyNr = threadIdx.x + blockIdx.x * blockDim.x;

    if (bodyNr >= data.numBodies)
        return;

    RigidBody body = data.bodies[bodyNr];
    Bounds bounds;
    bounds.minimum = +body.extent;
    bounds.maximum = -body.extent;
    bounds.transform(body.transform);

    data.boundsLowers[bodyNr] = Vec4(bounds.minimum);
    data.boundsUppers[bodyNr] = Vec4(bounds.maximum);
}


__device__ void device_createPairContactPoints(RigidBodyDeviceData data, int bodyNr0, int bodyNr1)
{
    RigidBody body0 = data.bodies[bodyNr0];
    RigidBody body1 = data.bodies[bodyNr1];

    // point face contacts

    // edge edge contacts

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

    

    
}

