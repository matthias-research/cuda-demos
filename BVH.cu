#include "BVH.h"
#include "CudaUtils.h"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

#include <vector>

#define FIXPOINT_SCALE 100.0f // two decimal places
#define FIXPOINT_INV_SCALE 0.01f
#define MAX_INT 2147483647
#define MORTON_BITS 30


struct BVHBuilderDeviceData
{
	void free() // no destructor because cuda would call it in the kernels
	{
		groupOfItem.free();
		indices.free();
		keys.free();
		longKeys.free();
		deltas.free();
		rangeLefts.free();
		rangeRights.free();
		numChildren.free();
		boundsOfGroups.free();
		numItems = 0;
		numGroups = 0;
	}

    size_t allocationSize() const
    {
        size_t s = 0;
		s += groupOfItem.allocationSize();
        s += indices.allocationSize();
		s += keys.allocationSize();
		s += longKeys.allocationSize();
		s += deltas.allocationSize();
		s += rangeLefts.allocationSize();
		s += rangeRights.allocationSize();
		s += numChildren.allocationSize();
		s += boundsOfGroups.allocationSize();
		return s;
	}

	int numItems = 0;
	int numGroups = 0;
	DeviceBuffer<int> groupOfItem; // derived from firstItemOfGroup if used
	DeviceBuffer<int> indices;
	DeviceBuffer<int> keys;
	DeviceBuffer<uint64_t> longKeys;
	DeviceBuffer<int> deltas;
	DeviceBuffer<int> rangeLefts;
	DeviceBuffer<int> rangeRights;
	DeviceBuffer<int> numChildren;
	DeviceBuffer<int> boundsOfGroups; // fixed point to be able to use integer atomicMin/Max, 6 values per bound
};

// -------------------------------------------------------------------------------------------
static void DeviceAlloc(void** ptr, size_t size)
{
	cudaCheck(cudaMalloc(ptr, size));
}

static void DeviceFree(void* ptr)
{
	cudaCheck(cudaFree(ptr));
}

void BVHBuilder::sortCellIndices(int* keys, int* values, int num, int numBits)
{
	thrust::device_ptr<int> d_keys(keys);
	thrust::device_ptr<int> d_values(values);
	
	thrust::sort_by_key(d_keys, d_keys + num, d_values);
}


void BVHBuilder::sortCellIndices(uint64_t* keys, int* values, int num, int numBits)
{
	thrust::device_ptr<uint64_t> d_keys(keys);
	thrust::device_ptr<int> d_values(values);
	
	thrust::sort_by_key(d_keys, d_keys + num, d_values);
}


CUDA_CALLABLE  inline int LongestAxis(const Vec3& v)
{
	if (v.x > v.y && v.x > v.z)
		return 0;
	if (v.y > v.z)
		return 1;
	else
		return 2;
}

CUDA_CALLABLE  inline int CLZ(int x)
{
	int n;
	if (x == 0) return 32;
	for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1);
	return n;
}

CUDA_CALLABLE  inline unsigned int Part1by2(unsigned int n)
{
	n = (n ^ (n << 16)) & 0xff0000ff;
	n = (n ^ (n <<  8)) & 0x0300f00f;
	n = (n ^ (n <<  4)) & 0x030c30c3;
	n = (n ^ (n <<  2)) & 0x09249249;

	return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*log2(Dim) bits
template <int Dim>
CUDA_CALLABLE  inline unsigned int Morton3(float x, float y, float z)
{
	unsigned int ux = Clamp(int(x*Dim), 0, Dim-1);
	unsigned int uy = Clamp(int(y*Dim), 0, Dim-1);
	unsigned int uz = Clamp(int(z*Dim), 0, Dim-1);

	return (Part1by2(uz) << 2) | (Part1by2(uy) << 1) | Part1by2(ux);
}

CUDA_CALLABLE  inline PackedNodeHalf MakeNode(const Vec3& bound, int child, bool leaf)
{
	PackedNodeHalf n;
	n.x = bound.x;
	n.y = bound.y;
	n.z = bound.z;
	n.i = (unsigned int)child;
	n.b = (unsigned int)(leaf?1:0);

	return n;
}

// variation of MakeNode through volatile pointers used in BuildHierarchy
CUDA_CALLABLE  inline void MakeNode(volatile PackedNodeHalf* n, const Vec3& bound, int child, bool leaf)
{
	n->x = bound.x;
	n->y = bound.y;
	n->z = bound.z;
	n->i = (unsigned int)child;
	n->b = (unsigned int)(leaf?1:0);
}

////////////////////////////////////////////////////////

__global__ void InitRoots(int* roots, int numRoots)
{
	const int rootNr = blockDim.x*blockIdx.x + threadIdx.x;

	if (rootNr < numRoots)
	{
		roots[rootNr] = -1; // -1 means no items in this group
	}
}

void BVHBuilder::resizeBVH(BVH& bvh, int numNodes, int numRoots)
{
	if (numNodes > bvh.mMaxNodes)
	{
		const int numToAlloc = int(numNodes*1.5f);

		DeviceFree(bvh.mNodeLowers);
		DeviceFree(bvh.mNodeUppers);

		DeviceAlloc((void**)&bvh.mNodeLowers, sizeof(PackedNodeHalf)*numToAlloc);
		DeviceAlloc((void**)&bvh.mNodeUppers, sizeof(PackedNodeHalf)*numToAlloc);

		bvh.mMaxNodes = numToAlloc;
	}
    if (numRoots > bvh.mMaxRoots)
    {
		const int numToAlloc = int(numRoots*1.5f);

		DeviceFree(bvh.mRootNodes);
		DeviceAlloc((void**)&bvh.mRootNodes, sizeof(int)*numToAlloc);

		bvh.mMaxRoots = numToAlloc;
    }

    bvh.mNumNodes = numNodes;
    bvh.mNumRoots = numRoots;

	InitRoots<<<numRoots / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(bvh.mRootNodes, numRoots);
}

void BVHBuilder::freeBVH(BVH& bvh)
{
	DeviceFree(bvh.mNodeLowers);
	DeviceFree(bvh.mNodeUppers);
    bvh.mNodeLowers = nullptr;
    bvh.mNodeUppers = nullptr;

	DeviceFree(bvh.mRootNodes);
    bvh.mRootNodes = nullptr;
    bvh.mMaxNodes = 0;
}

void BVHBuilder::cloneBVH(const BVH& hostBVH, BVH& deviceBVH)
{
	resizeBVH(deviceBVH, hostBVH.mMaxNodes, hostBVH.mMaxRoots);

	// copy host data to device
	cudaCheck(cudaMemcpy(deviceBVH.mNodeLowers, &hostBVH.mNodeLowers[0], sizeof(PackedNodeHalf)*hostBVH.mNumNodes, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(deviceBVH.mNodeUppers, &hostBVH.mNodeUppers[0], sizeof(PackedNodeHalf)*hostBVH.mNumNodes, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(deviceBVH.mRootNodes, &hostBVH.mRootNodes[0], sizeof(int)*hostBVH.mNumRoots, cudaMemcpyHostToDevice));
}

// build kernels

__global__ void InitGroupOfItem(BVHBuilderDeviceData data, const int* firstItemOfGroup)
{
    int groupNr = threadIdx.x + blockIdx.x * blockDim.x;
	if (groupNr == 0 || groupNr >= data.numGroups)
        return;

	int firstItem = firstItemOfGroup[groupNr];
	atomicAdd(&data.groupOfItem[firstItem], 1);
}


__global__ void InitBoundsOfGroups(BVHBuilderDeviceData data)
{
	const int groupNr = blockDim.x*blockIdx.x + threadIdx.x;

	if (groupNr < data.numGroups)
	{
		int* groupBounds = &data.boundsOfGroups[groupNr * 6];
        groupBounds[0] = MAX_INT; // min x
        groupBounds[1] = MAX_INT; // min y
        groupBounds[2] = MAX_INT; // min z
        groupBounds[3] = -MAX_INT; // max x
        groupBounds[4] = -MAX_INT; // max y
        groupBounds[5] = -MAX_INT; // max z
	}
}

__global__ void ComputeBoundsOfGroups(BVHBuilderDeviceData data, const Vec4* __restrict__ itemLowers, const Vec4* __restrict__ itemUppers)
{
	const int itemNr = blockDim.x*blockIdx.x + threadIdx.x;

	if (itemNr >= data.numItems)
		return;

    int groupNr = Clamp(data.groupOfItem[itemNr], 0, data.numGroups - 1);

	Vec4 lowers = itemLowers[itemNr];
	Vec4 uppers = itemUppers[itemNr];

	int* groupBounds = &data.boundsOfGroups[groupNr * 6];

	atomicMin(&groupBounds[0], int(lowers.x * FIXPOINT_SCALE) - 1);
    atomicMin(&groupBounds[1], int(lowers.y * FIXPOINT_SCALE) - 1);
    atomicMin(&groupBounds[2], int(lowers.z * FIXPOINT_SCALE) - 1);
    atomicMax(&groupBounds[3], int(uppers.x * FIXPOINT_SCALE) + 1);
    atomicMax(&groupBounds[4], int(uppers.y * FIXPOINT_SCALE) + 1);
    atomicMax(&groupBounds[5], int(uppers.z * FIXPOINT_SCALE) + 1);
}

__global__ void CalculateMortonCodes(BVHBuilderDeviceData data, const Vec4* __restrict__ itemLowers, const Vec4* __restrict__ itemUppers, Vec3 gridLower, Vec3 gridInvEdges)
{
	const int itemNr = blockDim.x*blockIdx.x + threadIdx.x;

	if (itemNr < data.numItems)
	{
		Vec3 lower = Vec3(itemLowers[itemNr].getXYZ());
		Vec3 upper = Vec3(itemUppers[itemNr].getXYZ());

		Vec3 center = 0.5f * (lower + upper);
		Vec3 local = (center - gridLower).multiply(gridInvEdges);

		int key = Morton3<1024>(local.x, local.y, local.z);

		data.indices[itemNr] = itemNr;
		data.keys[itemNr] = key;
	}
}

__global__ void CalculateGroupMortonCodes(BVHBuilderDeviceData data, const Vec4* __restrict__ itemLowers, const Vec4* __restrict__ itemUppers)
{
	const int itemNr = blockDim.x*blockIdx.x + threadIdx.x;

	if (itemNr < data.numItems)
	{
		Vec3 lower = Vec3(itemLowers[itemNr].getXYZ());
		Vec3 upper = Vec3(itemUppers[itemNr].getXYZ());

		Vec3 center = 0.5f * (lower + upper);
        int groupNr = Clamp(data.groupOfItem[itemNr], 0, data.numGroups - 1);

		// use the bounds of the group to calculate a local position

		int *intGroupBounds = &data.boundsOfGroups[groupNr * 6];
		Vec3 groupMin = Vec3((float)intGroupBounds[0], (float)intGroupBounds[1], (float)intGroupBounds[2]) * FIXPOINT_INV_SCALE;
		Vec3 groupMax = Vec3((float)intGroupBounds[3], (float)intGroupBounds[4], (float)intGroupBounds[5]) * FIXPOINT_INV_SCALE;

        Vec3 edges = groupMax - groupMin;
        edges += Vec3(0.0001f, 0.0001f, 0.0001f);
        Vec3 invEdges = Vec3(1.0f / edges.x, 1.0f / edges.y, 1.0f / edges.z);
		Vec3 local = (center - groupMin).multiply(invEdges);

		int morton = Morton3<1024>(local.x, local.y, local.z);
		uint64_t key = (uint64_t(groupNr) << MORTON_BITS) | uint64_t(morton);

		data.indices[itemNr] = itemNr;
		data.longKeys[itemNr] = key;
	}
}

// calculate the index of the first differing bit between two adjacent Morton keys
__global__ void CalculateKeyDeltas(BVHBuilderDeviceData data, bool useGroups)
{
	const int itemNr = blockDim.x*blockIdx.x + threadIdx.x;

	if (itemNr < data.numItems - 1)
	{
		int x = 0;

		if (useGroups)
		{
			int a = (int)(data.longKeys[itemNr] & 0xFFFFFFFF);
			int b = (int)(data.longKeys[itemNr + 1] & 0xFFFFFFFF);
			x = a^b;
		}
		else
		{
			int a = data.keys[itemNr];
			int b = data.keys[itemNr + 1];
			x = a^b;
		}

		data.deltas[itemNr] = x;// __clz(x);
	}
}

__global__ void BuildLeaves(BVHBuilderDeviceData data, const Vec4* __restrict__ itemLowers, const Vec4* __restrict__ itemUppers,
    PackedNodeHalf* __restrict__ lowers, PackedNodeHalf* __restrict__ uppers, const int* __restrict__ firstItemOfGroup)
{
	const int index = blockDim.x*blockIdx.x + threadIdx.x;

	if (index < data.numItems)
	{
		int itemNr = data.indices[index];
        int localItemNr = itemNr;

		const Vec3 lower = Vec3(itemLowers[itemNr].getXYZ());
		const Vec3 upper = Vec3(itemUppers[itemNr].getXYZ());

		// write leaf nodes

        if (firstItemOfGroup)
        {
            int groupNr = data.groupOfItem[itemNr];
            localItemNr = itemNr - firstItemOfGroup[groupNr];
        }

		lowers[index] = MakeNode(lower, itemNr, true);
		uppers[index] = MakeNode(upper, localItemNr, false); // the item number relative to the beginning of the group

		// write leaf key ranges
		data.rangeLefts[index] = index;
		data.rangeRights[index] = index;
	}
}

// this bottom-up process assigns left and right children and combines bounds to form internal nodes
// there is one thread launched per-leaf node, each thread calculates it's parent node and assigns
// itself to either the left or right parent slot, the last child to complete the parent and moves
// up the hierarchy

__global__ void BuildHierarchy(BVHBuilderDeviceData data, int* roots, volatile PackedNodeHalf* __restrict__ lowers, volatile PackedNodeHalf* __restrict__ uppers, bool useGroups)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < data.numItems)
	{
		const int internalOffset = data.numItems;

    	int thisGroup = 0;
        if (useGroups)
        {
            thisGroup = (int)(data.longKeys[index] >> MORTON_BITS);
        }

		for (;;)
		{
			int left = data.rangeLefts[index];
			int right = data.rangeRights[index];

			bool leftAtBoundary = (left == 0);
			bool rightAtBoundary = (right == data.numItems - 1);

			if (useGroups)
			{
                int leftIdx = Max(0, left - 1);
                int rightIdx = Min(right + 1, data.numItems - 1);

                int leftGroup = (int)(data.longKeys[leftIdx] >> MORTON_BITS);
                int rightGroup = (int)(data.longKeys[rightIdx] >> MORTON_BITS);

                bool leftGroupChange = leftGroup != thisGroup;
                bool rightGroupChange = rightGroup != thisGroup;

                leftAtBoundary = leftAtBoundary || leftGroupChange;
                rightAtBoundary = rightAtBoundary || rightGroupChange;
            }

			// check if we are the root node, if so then store out our index and terminate
			if (leftAtBoundary && rightAtBoundary)
			{
				roots[thisGroup] = index;
				break;
			}

			int childCount = 0;

			int parent;

			if (leftAtBoundary || (!rightAtBoundary && data.deltas[right] < data.deltas[left - 1]))
			{
                parent = right + internalOffset;

				// set parent left child
				lowers[parent].i = index;
				data.rangeLefts[parent] = left;

				// ensure above writes are visible to all threads
				__threadfence();

				childCount = atomicAdd(&data.numChildren[parent], 1);
			}
			else
			{
                parent = left + internalOffset - 1;

				// set parent right child
				uppers[parent].i = index;
				data.rangeRights[parent] = right;

				// ensure above writes are visible to all threads
				__threadfence();

				childCount = atomicAdd(&data.numChildren[parent], 1);
			}

			// if we have are the last thread (such that the parent node is now complete)
			// then update its bounds and move onto the the next parent in the hierarchy
			if (childCount == 1)
			{
				const int leftChild = lowers[parent].i;
				const int rightChild = uppers[parent].i;

				Vec3 leftLower = Vec3(lowers[leftChild].x,
					lowers[leftChild].y,
					lowers[leftChild].z);

				Vec3 leftUpper = Vec3(uppers[leftChild].x,
					uppers[leftChild].y,
					uppers[leftChild].z);

				Vec3 rightLower = Vec3(lowers[rightChild].x,
					lowers[rightChild].y,
					lowers[rightChild].z);


				Vec3 rightUpper = Vec3(uppers[rightChild].x,
					uppers[rightChild].y,
					uppers[rightChild].z);

				// union of child bounds
				Vec3 lower = leftLower.minimum(rightLower);
				Vec3 upper = leftUpper.maximum(rightUpper);

				// write new BVH nodes
				MakeNode(lowers + parent, lower, leftChild, false);
				MakeNode(uppers + parent, upper, rightChild, false);

				// move onto processing the parent
				index = parent;
			}
			else
			{
				// parent not ready (we are the first child), terminate thread
				break;
			}
		}
	}
}

BVHBuilder::BVHBuilder()
{
	mDeviceData = new BVHBuilderDeviceData();
}

BVHBuilder::~BVHBuilder()
{
    free();
	delete mDeviceData;
}

// ---------------------------------------------------------------------------
void BVHBuilder::free()
{
	if (mDeviceData)
	{
		mDeviceData->free();
	}
}

// ---------------------------------------------------------------------------
size_t BVHBuilder::allocationSize()
{
    if (mDeviceData)
    {
        return mDeviceData->allocationSize();
    }
    else
    {
        return 0;
    }
}

// ---------------------------------------------------------------------------

struct CompareX
{
    __host__ __device__ bool operator()(const Vec4& a, const Vec4& b) const
    {
        return a.x < b.x;
    }
};

struct CompareY
{
    __host__ __device__ bool operator()(const Vec4& a, const Vec4& b) const
    {
        return a.y < b.y;
    }
};

struct CompareZ
{
    __host__ __device__ bool operator()(const Vec4& a, const Vec4& b) const
    {
        return a.z < b.z;
    }
};

static Bounds3 computeBounds(const Vec4* points, int numPoints)
{
    thrust::device_ptr<const Vec4> thrustPoints(points);
    auto minmaxX = thrust::minmax_element(thrust::device, thrustPoints, thrustPoints + numPoints, CompareX());
    auto minmaxY = thrust::minmax_element(thrust::device, thrustPoints, thrustPoints + numPoints, CompareY());
    auto minmaxZ = thrust::minmax_element(thrust::device, thrustPoints, thrustPoints + numPoints, CompareZ());

    Vec4 minX, maxX, minY, maxY, minZ, maxZ;
    thrust::copy(minmaxX.first, minmaxX.first + 1, &minX);
    thrust::copy(minmaxX.second, minmaxX.second + 1, &maxX);
    thrust::copy(minmaxY.first, minmaxY.first + 1, &minY);
    thrust::copy(minmaxY.second, minmaxY.second + 1, &maxY);
    thrust::copy(minmaxZ.first, minmaxZ.first + 1, &minZ);
    thrust::copy(minmaxZ.second, minmaxZ.second + 1, &maxZ);
    return Bounds3(Vec3(minX.x, minY.y, minZ.z), Vec3(maxX.x, maxY.y, maxZ.z));
}


// ---------------------------------------------------------------------------
void BVHBuilder::build(BVH& bvh,
                          const Vec4* itemLowers,
                          const Vec4* itemUppers,
                          int numItems,
                          const int* firstItemOfGroup,
                          int numGroups)
{
	mDeviceData->numItems = numItems;
	mDeviceData->numGroups = numGroups;
	const int maxNodes = 2 * numItems;

	resizeBVH(bvh, maxNodes, Max(numGroups, 1));

	bool useGroups = firstItemOfGroup != nullptr;

	// compute total bounds

	Bounds3 totalBounds(Empty);

	if (useGroups)
	{
		// compute the group of each item from the firstItemOfGroup array

		mDeviceData->groupOfItem.resize(numItems, false);
		mDeviceData->groupOfItem.setZero();
		InitGroupOfItem<<<numGroups / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*mDeviceData, firstItemOfGroup);
        thrust::device_ptr<int> groupOfItem(mDeviceData->groupOfItem.buffer);
        thrust::inclusive_scan(groupOfItem, groupOfItem + numItems, groupOfItem);

		// compute bounds for each group

		mDeviceData->boundsOfGroups.resize(6 * numGroups, false);
		InitBoundsOfGroups<<<numGroups / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*mDeviceData);
		ComputeBoundsOfGroups<<<numItems / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*mDeviceData, itemLowers, itemUppers);
	}
	else
	{
		totalBounds = computeBounds(itemLowers, numItems);
		Bounds3 upperBounds = computeBounds(itemUppers, numItems);
		totalBounds.include(upperBounds);
	}


	mDeviceData->indices.resize(numItems * 2, false);	// *2 for radix sort
	mDeviceData->deltas.resize(numItems, false);	// highest differenting bit between keys for item i and i+1
	mDeviceData->rangeLefts.resize(maxNodes, false);
    mDeviceData->rangeLefts.setZero();
	mDeviceData->rangeRights.resize(maxNodes, false);
    mDeviceData->rangeRights.setZero();
	mDeviceData->numChildren.resize(maxNodes, false);

	// calculate Morton codes
	// assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf

	if (useGroups)
	{
		mDeviceData->longKeys.resize(numItems * 2, false);	// *2 for radix sort
		CalculateGroupMortonCodes<<<numItems / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*mDeviceData, itemLowers, itemUppers);
		int numGroupBits = 1;
		while ((1 << numGroupBits) < numGroups)
			numGroupBits++;

		sortCellIndices(mDeviceData->longKeys.buffer, mDeviceData->indices.buffer, numItems, numGroupBits + MORTON_BITS);
	}
    else
    {
        mDeviceData->keys.resize(numItems * 2, false); // *2 for radix sort
        Vec3 edges = totalBounds.getDimensions();
        edges += Vec3(0.0001f, 0.0001f, 0.0001f);
        Vec3 invEdges = Vec3(1.0f / edges.x, 1.0f / edges.y, 1.0f / edges.z);

        CalculateMortonCodes<<<numItems / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(
            *mDeviceData, itemLowers, itemUppers, totalBounds.minimum, invEdges);
        sortCellIndices(mDeviceData->keys.buffer, mDeviceData->indices.buffer, numItems, MORTON_BITS);
    }

	// calculate deltas between adjacent keys
	CalculateKeyDeltas<<<numItems / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*mDeviceData, useGroups);

	// initialize leaf nodes
    BuildLeaves<<<numItems / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(*mDeviceData,
        itemLowers, itemUppers, bvh.mNodeLowers, bvh.mNodeUppers, firstItemOfGroup);

	// reset children count, this is our atomic counter so we know when an internal node is complete, only used during building
    mDeviceData->numChildren.setZero();

	// build the tree and internal node bounds
    BuildHierarchy<<<numItems / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(
        *mDeviceData, bvh.mRootNodes, bvh.mNodeLowers, bvh.mNodeUppers, useGroups);

}

