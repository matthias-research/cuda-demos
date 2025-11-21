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

CUDA_CALLABLE inline Vec3 getClosestPointOnTriangle(
    const Vec3& p, const Vec3& p0, const Vec3& p1, const Vec3& p2, bool* inside = nullptr)
{
    Vec3 e0 = p1 - p0;
    Vec3 e1 = p2 - p0;
    Vec3 tmp = p0 - p;

    float a = e0.dot(e0);
    float b = e0.dot(e1);
    float c = e1.dot(e1);
    float d = e0.dot(tmp);
    float e = e1.dot(tmp);
    Vec3 coords(b * e - c * d, b * d - a * e, a * c - b * b);

    float x = 0.0f;
    float y = 0.0f;
    if (inside)
    {
        *inside = false;
    }
    if (coords[0] <= 0.0f)
    {
        if (c != 0.0f)
            y = -e / c;
    }
    else if (coords[1] <= 0.0f)
    {
        if (a != 0.0f)
            x = -d / a;
    }
    else if (coords[0] + coords[1] > coords[2])
    {
        float den = a + c - b - b;
        float num = c + e - b - d;
        if (den != 0.0f)
        {
            x = num / den;
            y = 1.0f - x;
        }
    }
    else
    {
        if (coords[2] != 0.0f)
        {
            x = coords[0] / coords[2];
            y = coords[1] / coords[2];
        }
        if (inside)
        {
            *inside = true;
        }
    }

    x = Clamp(x, 0.0f, 1.0f);
    y = Clamp(y, 0.0f, 1.0f);

    return Vec3(1.0f - x - y, x, y);
}
