#pragma once

#include "Vec.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

struct PackedNodeHalf
{
    float x;
    float y;
    float z;
    unsigned int i : 31;
    unsigned int b : 1;
};

struct BVH
{
    BVH() : mRootNodes(nullptr), mNodeLowers(nullptr), mNodeUppers(nullptr), mNumNodes(0), mMaxNodes(0), mMaxDepth(0)
    {
    }

    // The entries hold the indices of the roots of the BVH trees
    // If no groups are used, there is a single root node
    // otherwise one root node per group
    int* mRootNodes;

    PackedNodeHalf* __restrict__ mNodeLowers; // x, y, z are the lower spatial bounds of the node's children
                                              // for internal nodes b is zero and i is a pointher the the left child
                                              // node for leaf nodes b is non zero and i is the *global* number of the
                                              // item

    PackedNodeHalf* __restrict__ mNodeUppers; // x, y, z are the upper spatial bounds of the node's children
                                              // for internal nodes i is a pointer to the right child node
                                              // for leaf nodes i is the *local* number of the item with respect to the
                                              // group start b is not used

    int mNumNodes = 0;
    int mMaxNodes = 0;
    int mNumRoots = 0;
    int mMaxRoots = 0;
    int mMaxDepth = 0;

    void free()
    {
        if (mRootNodes)
        {
            cudaFree(mRootNodes);
            mRootNodes = nullptr;
        }
        if (mNodeLowers)
        {
            cudaFree(mNodeLowers);
            mNodeLowers = nullptr;
        }
        if (mNodeUppers)
        {
            cudaFree(mNodeUppers);
            mNodeUppers = nullptr;
        }
        mMaxNodes = 0;
        mNumNodes = 0;
        mMaxRoots = 0;
        mNumRoots = 0;
        mMaxDepth = 0;
    }


    size_t size() const
    {
        return sizeof(PackedNodeHalf) * mMaxNodes * 2 + mNumRoots * sizeof(int);
    }
};


struct BVHBuilderDeviceData;

class BVHBuilder
{
public:
    BVHBuilder();
    ~BVHBuilder();

    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each item
    // for multiple groups, firstItemOfGroup defines the position of the first item of each group,
    // if nullptr, a single group is assumed

    void build(BVH& bvh,
               const Vec::Vec4* itemLowers,
               const Vec::Vec4* itemUppers,
               int numItems,
               const int* firstItemOfGroup = nullptr,
               int numGroups = 1);

    static void resizeBVH(BVH& bvh, int numNodes, int numRoots = 1);
    static void freeBVH(BVH& bvh);
    static void cloneBVH(const BVH& hostBVH, BVH& deviceBVH);
    void free();
    size_t allocationSize();

    void sortCellIndices(int* keys, int* values, int num, int numBits);
    void sortCellIndices(uint64_t* keys, int* values, int num, int numBits);

private:
    // temporary data used during building

    BVHBuilderDeviceData* mDeviceData = nullptr;
};

