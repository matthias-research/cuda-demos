#include "Scene.h"
#include "Vec.h"
#include <memory>

class MeshesDeviceData;
class BVHBuilder;

class CudaMeshes
{
public:
    CudaMeshes() = default;
    ~CudaMeshes() = default;

    void initialize(const Scene* scene);
    void cleanup();

    bool rayCast(const Ray& ray, float& minT);
    void handleCollisions(int numParticles, float* positions, int stride, 
           float* radii, float defaultRadius, float searchDist);

private:
    std::shared_ptr<MeshesDeviceData> deviceData = nullptr;
    std::shared_ptr<BVHBuilder> bvhBuilder = nullptr;
};

