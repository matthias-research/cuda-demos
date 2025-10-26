#pragma once

#include "Vec.h"

CUDA_CALLABLE inline bool getClosestPointsOnRays(const Ray& ray0, const Ray& ray1, float& t0, float& t1)
{
    float a = ray0.dir.magnitudeSquared();
    float b = -ray0.dir.dot(ray1.dir);
    float c = ray0.dir.dot(ray1.dir);
    float d = -ray1.dir.magnitudeSquared();
    float e = (ray1.orig - ray0.orig).dot(ray0.dir);
    float f = (ray1.orig - ray0.orig).dot(ray1.dir);
    float det = a * d - b * c;
    if (det == 0.0f) // rays are parallel
    {
        t0 = 0.0f; // arbitrary
        t1 = 0.0f;
        return false;
    }
    det = 1.0f / det;
    t0 = (e * d - b * f) * det;
    t1 = (a * f - e * c) * det;
    return true;
}