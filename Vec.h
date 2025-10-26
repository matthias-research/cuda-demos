#pragma once

#include <math.h>
#include <vector>

#ifdef __CUDACC__
#    define CUDA_CALLABLE __host__ __device__
#    define CUDA_CALLABLE_DEVICE __device__
#else
#    define CUDA_CALLABLE
#    define CUDA_CALLABLE_DEVICE
#endif

#ifndef UNUSED
#    define UNUSED(x) (void)(x)
#endif


// ----------------------------------------------------------------------------
#define Pi 3.141592653589793f
#define TwoPi (2.0f * Pi)
#define HalfPi (0.5f * Pi)
#define MaxFloat 3.402823466e+38f
#define MaxInt32 0x7fffffff

enum _ZERO
{
    Zero
};
enum _IDENTITY
{
    Identity
};
enum _EMPTY
{
    Empty
};

CUDA_CALLABLE inline float Min(float f0, float f1)
{
    return f0 < f1 ? f0 : f1;
}

CUDA_CALLABLE inline float Max(float f0, float f1)
{
    return f0 > f1 ? f0 : f1;
}

CUDA_CALLABLE inline float Min(float f0, float f1, float f2)
{
    return Min(Min(f0, f1), f2);
}

CUDA_CALLABLE inline float Max(float f0, float f1, float f2)
{
    return Max(Max(f0, f1), f2);
}

CUDA_CALLABLE inline int Min(int i0, int i1)
{
    return i0 < i1 ? i0 : i1;
}

CUDA_CALLABLE inline int Max(int f0, int f1)
{
    return f0 > f1 ? f0 : f1;
}

CUDA_CALLABLE inline size_t Min(size_t i0, size_t i1)
{
    return i0 < i1 ? i0 : i1;
}

CUDA_CALLABLE inline size_t Max(size_t f0, size_t f1)
{
    return f0 > f1 ? f0 : f1;
}

CUDA_CALLABLE inline float Abs(float f)
{
    return f >= 0.0f ? f : -f;
}

CUDA_CALLABLE inline int Abs(int i)
{
    return i >= 0 ? i : -i;
}

/**
 * @brief Clamps a floating point value to the range [a, b].
 * @param f The value to clamp.
 * @param a The minimum value.
 * @param b The maximum value.
 * @return The clamped value.
 */
CUDA_CALLABLE inline float Clamp(float f, float a, float b)
{
    return Min(b, Max(a, f));
}

/**
 * @brief Clamps an integer value to the range [a, b].
 * @param f The value to clamp.
 * @param a The minimum value.
 * @param b The maximum value.
 * @return The clamped value.
 */
CUDA_CALLABLE inline int Clamp(int f, int a, int b)
{
    return Min(b, Max(a, f));
}

/**
 * @brief Computes the sine of an angle in radians.
 */
CUDA_CALLABLE inline float Sin(float f)
{
    return sinf(f);
}

/**
 * @brief Computes the cosine of an angle in radians.
 */
CUDA_CALLABLE inline float Cos(float f)
{
    return cosf(f);
}

/**
 * @brief Computes the arc sine of a value safely clamping the input to the range [-1, 1].
 */
CUDA_CALLABLE inline float Asin(float f)
{
    return asinf(Clamp(f, -1.0f, 1.0f));
}

/**
 * @brief Computes the arc cosine of a value safely clamping the input to the range [-1, 1].
 */
CUDA_CALLABLE inline float Acos(float f)
{
    return acosf(Clamp(f, -1.0f, 1.0f));
}

/**
 * @brief Rounds a floating point value to a specified number of digits.
 */
CUDA_CALLABLE inline float Round(float f, int numDigits)
{
    float p = powf(10.0f, (float)numDigits);
    return f > 0.0f ? floorf(p * f + 0.5f) / p : ceilf(p * f - 0.5f) / p;
}

/**
 * @brief Defines a two dimensional vector with the float components x and y.
 */
struct Vec2
{
    CUDA_CALLABLE Vec2()
    {
    }
    CUDA_CALLABLE Vec2(_ZERO) : x(0.0f), y(0.0f)
    {
    }
    CUDA_CALLABLE Vec2(float x, float y) : x(x), y(y)
    {
    }
    CUDA_CALLABLE Vec2(float* f)
    {
        x = f[0];
        y = f[1];
    }
    CUDA_CALLABLE Vec2(_IDENTITY, int dim)
    {
        x = dim == 0 ? 1.0f : 0.0f;
        y = dim == 1 ? 1.0f : 0.0f;
    }

    CUDA_CALLABLE float& operator[](unsigned int index)
    {
        return reinterpret_cast<float*>(this)[index];
    }

    CUDA_CALLABLE const float& operator[](unsigned int index) const
    {
        return reinterpret_cast<const float*>(this)[index];
    }

    CUDA_CALLABLE bool operator==(const Vec2& v) const
    {
        return x == v.x && y == v.y;
    }
    CUDA_CALLABLE bool operator!=(const Vec2& v) const
    {
        return x != v.x || y != v.y;
    }

    CUDA_CALLABLE bool operator<(const Vec2& v) const
    { // order dependent!
        if (x < v.x)
            return true;
        if (x > v.x)
            return false;
        return y < v.y;
    }

    CUDA_CALLABLE bool isZero() const
    {
        return x == 0.0 && y == 0.0;
    }

    CUDA_CALLABLE void set(float newX, float newY)
    {
        x = newX;
        y = newY;
    }

    CUDA_CALLABLE Vec2 minimum(const Vec2& v) const
    {
        return Vec2(Min(x, v.x), Min(y, v.y));
    }

    CUDA_CALLABLE Vec2 maximum(const Vec2& v) const
    {
        return Vec2(Max(x, v.x), Max(y, v.y));
    }

    CUDA_CALLABLE Vec2 lerp(const Vec2& v, float s) const
    {
        return *this * (1.0f - s) + v * s;
    }

    CUDA_CALLABLE Vec2 normal() const
    {
        return Vec2(y, -x);
    }

    CUDA_CALLABLE Vec2 operator-() const
    {
        return Vec2(-x, -y);
    }

    CUDA_CALLABLE Vec2 operator+(const Vec2& v) const
    {
        return Vec2(x + v.x, y + v.y);
    }

    CUDA_CALLABLE Vec2 operator-(const Vec2& v) const
    {
        return Vec2(x - v.x, y - v.y);
    }

    CUDA_CALLABLE void operator+=(const Vec2& v)
    {
        x += v.x;
        y += v.y;
    }

    CUDA_CALLABLE void operator-=(const Vec2& v)
    {
        x -= v.x;
        y -= v.y;
    }

    CUDA_CALLABLE void operator*=(float f)
    {
        x *= f;
        y *= f;
    }

    CUDA_CALLABLE void operator/=(float f)
    {
        x /= f;
        y /= f;
    }

    CUDA_CALLABLE Vec2 operator*(float f) const
    {
        return Vec2(x * f, y * f);
    }

    CUDA_CALLABLE Vec2 operator/(float f) const
    {
        return Vec2(x / f, y / f);
    }

    CUDA_CALLABLE Vec2 multiply(const Vec2& a) const
    {
        return Vec2(x * a.x, y * a.y);
    }

    CUDA_CALLABLE Vec2 diagonalMult(const Vec2& v) const
    {
        return Vec2(x * v.x, y * v.y);
    }

    CUDA_CALLABLE void setMin(const Vec2& v)
    {
        x = x < v.x ? x : v.x;
        y = y < v.y ? y : v.y;
    }

    CUDA_CALLABLE void setMax(const Vec2& v)
    {
        x = x > v.x ? x : v.x;
        y = y > v.y ? y : v.y;
    }

    CUDA_CALLABLE float magnitude() const
    {
        return sqrtf(x * x + y * y);
    }

    CUDA_CALLABLE float magnitudeSquared() const
    {
        return x * x + y * y;
    }

    CUDA_CALLABLE float length() const
    {
        return sqrtf(x * x + y * y);
    }

    CUDA_CALLABLE float lengthSquared() const
    {
        return x * x + y * y;
    }

    CUDA_CALLABLE float normalize()
    {
        float m = magnitude();
        if (m > 0.0f)
            *this /= m;
        return m;
    }

    CUDA_CALLABLE Vec2 normalized() const
    {
        Vec2 v = *this;
        v.normalize();
        return v;
    }

    CUDA_CALLABLE float dot(const Vec2& v) const
    {
        return x * v.x + y * v.y;
    }

    CUDA_CALLABLE float cross(const Vec2& v) const
    {
        return x * v.y - y * v.x;
    }

    CUDA_CALLABLE Vec2 perp() const
    {
        return Vec2(-y, x);
    }

    float x, y;
};

CUDA_CALLABLE inline Vec2 operator*(float f, const Vec2& v)
{
    return Vec2(f * v.x, f * v.y);
}

/**
 * @brief Defines a three dimensional vector with the float components x, y and z.
 */
struct Vec3
{
    CUDA_CALLABLE Vec3()
    {
    }
    CUDA_CALLABLE Vec3(const Vec2& v) : x(v.x), y(v.y), z(0.0)
    {
    }
    CUDA_CALLABLE Vec3(_ZERO) : x(0.0f), y(0.0f), z(0.0f)
    {
    }
    CUDA_CALLABLE Vec3(float x, float y, float z) : x(x), y(y), z(z)
    {
    }
    CUDA_CALLABLE Vec3(float* f)
    {
        x = f[0];
        y = f[1];
        z = f[2];
    }
    CUDA_CALLABLE Vec3(_IDENTITY, int dim)
    {
        x = dim == 0 ? 1.0f : 0.0f;
        y = dim == 1 ? 1.0f : 0.0f;
        z = dim == 2 ? 1.0f : 0.0f;
    }

    CUDA_CALLABLE float& operator[](unsigned int index)
    {
        return reinterpret_cast<float*>(this)[index];
    }

    CUDA_CALLABLE const float& operator[](unsigned int index) const
    {
        return reinterpret_cast<const float*>(this)[index];
    }

    CUDA_CALLABLE bool operator==(const Vec3& v) const
    {
        return x == v.x && y == v.y && z == v.z;
    }
    CUDA_CALLABLE bool operator!=(const Vec3& v) const
    {
        return x != v.x || y != v.y || z != v.z;
    }

    CUDA_CALLABLE bool operator<(const Vec3& v) const
    { // order dependent!
        if (x < v.x)
            return true;
        if (x > v.x)
            return false;
        if (y < v.y)
            return true;
        if (y > v.y)
            return false;
        return z < v.z;
    }

    CUDA_CALLABLE bool isZero() const
    {
        return x == 0.0 && y == 0.0 && z == 0.0;
    }

    CUDA_CALLABLE void set(float newX, float newY, float newZ)
    {
        x = newX;
        y = newY;
        z = newZ;
    }

    CUDA_CALLABLE Vec3 minimum(const Vec3& v) const
    {
        return Vec3(Min(x, v.x), Min(y, v.y), Min(z, v.z));
    }

    CUDA_CALLABLE Vec3 maximum(const Vec3& v) const
    {
        return Vec3(Max(x, v.x), Max(y, v.y), Max(z, v.z));
    }

    CUDA_CALLABLE Vec3 lerp(const Vec3& v, float s) const
    {
        return *this * (1.0f - s) + v * s;
    }

    CUDA_CALLABLE Vec3 operator-() const
    {
        return Vec3(-x, -y, -z);
    }

    CUDA_CALLABLE Vec3 operator+(const Vec3& v) const
    {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    CUDA_CALLABLE Vec3 operator-(const Vec3& v) const
    {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    CUDA_CALLABLE void operator+=(const Vec3& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    CUDA_CALLABLE void operator-=(const Vec3& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    CUDA_CALLABLE void operator*=(float f)
    {
        x *= f;
        y *= f;
        z *= f;
    }

    CUDA_CALLABLE void operator/=(float f)
    {
        x /= f;
        y /= f;
        z /= f;
    }

    CUDA_CALLABLE Vec3 operator*(float f) const
    {
        return Vec3(x * f, y * f, z * f);
    }

    CUDA_CALLABLE Vec3 operator/(float f) const
    {
        return Vec3(x / f, y / f, z / f);
    }

    CUDA_CALLABLE Vec3 multiply(const Vec3& a) const
    {
        return Vec3(x * a.x, y * a.y, z * a.z);
    }

    CUDA_CALLABLE Vec3 diagonalMult(const Vec3& v) const
    {
        return Vec3(x * v.x, y * v.y, z * v.z);
    }

    CUDA_CALLABLE void setMin(const Vec3& v)
    {
        x = x < v.x ? x : v.x;
        y = y < v.y ? y : v.y;
        z = z < v.z ? z : v.z;
    }

    CUDA_CALLABLE void setMax(const Vec3& v)
    {
        x = x > v.x ? x : v.x;
        y = y > v.y ? y : v.y;
        z = z > v.z ? z : v.z;
    }

    CUDA_CALLABLE float magnitude() const
    {
        return sqrtf(x * x + y * y + z * z);
    }

    CUDA_CALLABLE float magnitudeSquared() const
    {
        return x * x + y * y + z * z;
    }

    CUDA_CALLABLE float length() const
    {
        return sqrtf(x * x + y * y + z * z);
    }

    CUDA_CALLABLE float lengthSquared() const
    {
        return x * x + y * y + z * z;
    }

    CUDA_CALLABLE float normalize()
    {
        float m = magnitude();
        if (m > 0.0f)
            *this /= m;
        return m;
    }

    CUDA_CALLABLE Vec3 normalized() const
    {
        Vec3 v = *this;
        v.normalize();
        return v;
    }

    CUDA_CALLABLE float dot(const Vec3& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    CUDA_CALLABLE Vec3 cross(const Vec3& v) const
    {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    CUDA_CALLABLE Vec3 perp() const
    {
        if (isZero())
            return Vec3(1.0, 0.0, 0.0);
        Vec3 a(0.0, 0.0, 1.0);
        if (fabs(x) < fabs(y) && fabs(x) < fabs(z))
            a = Vec3(1.0, 0.0, 0.0);
        else if (fabs(y) < fabs(z))
            a = Vec3(0.0, 1.0, 0.0);
        Vec3 b = a.cross(*this);
        b.normalize();
        return b;
    }

    float x, y, z;
};

CUDA_CALLABLE inline Vec3 operator*(float f, const Vec3& v)
{
    return Vec3(f * v.x, f * v.y, f * v.z);
}

/**
 * @brief Defines a four dimensional vector with the float components x, y, z and w.
 */
struct Vec4
{
    CUDA_CALLABLE Vec4()
    {
    }
    CUDA_CALLABLE Vec4(_ZERO) : x(0.0f), y(0.0f), z(0.0f), w(0.0f)
    {
    }
    CUDA_CALLABLE Vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w)
    {
    }
    CUDA_CALLABLE Vec4(const Vec3& v, float w = 0.0f) : x(v.x), y(v.y), z(v.z), w(w)
    {
    }
    CUDA_CALLABLE Vec4(float* f)
    {
        x = f[0];
        y = f[1];
        z = f[2];
        w = f[3];
    }
    CUDA_CALLABLE Vec4(_IDENTITY, int dim)
    {
        x = dim == 0 ? 1.0f : 0.0f;
        y = dim == 1 ? 1.0f : 0.0f;
        z = dim == 2 ? 1.0f : 0.0f;
        w = dim == 3 ? 1.0f : 0.0f;
    }
    CUDA_CALLABLE Vec3 getXYZ() const
    {
        return Vec3(x, y, z);
    }

    CUDA_CALLABLE float& operator[](unsigned int index)
    {
        return reinterpret_cast<float*>(this)[index];
    }

    CUDA_CALLABLE const float& operator[](unsigned int index) const
    {
        return reinterpret_cast<const float*>(this)[index];
    }

    CUDA_CALLABLE bool isZero() const
    {
        return x == 0.0 && y == 0.0 && z == 0.0 && w == 0.0;
    }

    CUDA_CALLABLE void set(float newX, float newY, float newZ, float newW)
    {
        x = newX;
        y = newY;
        z = newZ;
        w = newW;
    }

    CUDA_CALLABLE Vec4 operator-() const
    {
        return Vec4(-x, -y, -z, -w);
    }

    CUDA_CALLABLE Vec4 operator+(const Vec4& v) const
    {
        return Vec4(x + v.x, y + v.y, z + v.z, w + v.w);
    }

    CUDA_CALLABLE Vec4 operator-(const Vec4& v) const
    {
        return Vec4(x - v.x, y - v.y, z - v.z, w - v.w);
    }

    CUDA_CALLABLE void operator+=(const Vec4& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
    }

    CUDA_CALLABLE void operator-=(const Vec4& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
    }

    CUDA_CALLABLE void operator*=(float f)
    {
        x *= f;
        y *= f;
        z *= f;
        w *= f;
    }

    CUDA_CALLABLE void operator/=(float f)
    {
        x /= f;
        y /= f;
        z /= f;
        w /= f;
    }

    CUDA_CALLABLE Vec4 operator*(float f) const
    {
        return Vec4(x * f, y * f, z * f, w * f);
    }

    CUDA_CALLABLE Vec4 operator/(float f) const
    {
        return Vec4(x / f, y / f, z / f, w / f);
    }

    CUDA_CALLABLE bool operator==(const Vec4& v) const
    {
        return x == v.x && y == v.y && z == v.z && w == v.w;
    }

    CUDA_CALLABLE void setMin(const Vec4& v)
    {
        x = x < v.x ? x : v.x;
        y = y < v.y ? y : v.y;
        z = z < v.z ? z : v.z;
        w = w < v.w ? w : v.w;
    }

    CUDA_CALLABLE void setMax(const Vec4& v)
    {
        x = x > v.x ? x : v.x;
        y = y > v.y ? y : v.y;
        z = z > v.z ? z : v.z;
        w = w > v.w ? w : v.w;
    }

    CUDA_CALLABLE float magnitude() const
    {
        return sqrtf(x * x + y * y + z * z + w * w);
    }

    CUDA_CALLABLE float magnitudeSquared() const
    {
        return x * x + y * y + z * z + w * w;
    }

    CUDA_CALLABLE float length() const
    {
        return sqrtf(x * x + y * y + z * z + w * w);
    }

    CUDA_CALLABLE float lengthSquared() const
    {
        return x * x + y * y + z * z + w * w;
    }

    CUDA_CALLABLE float normalize()
    {
        float m = magnitude();
        if (m > 0.0f)
            *this /= m;
        return m;
    }

    CUDA_CALLABLE Vec4 normalized() const
    {
        Vec4 v = *this;
        v.normalize();
        return v;
    }

    CUDA_CALLABLE float dot(const Vec4& v) const
    {
        return x * v.x + y * v.y + z * v.z + w * v.w;
    }

    float x, y, z, w;
};

CUDA_CALLABLE inline Vec4 operator*(float f, const Vec4& v)
{
    return Vec4(f * v.x, f * v.y, f * v.z, f * v.w);
}


CUDA_CALLABLE inline Vec2 Clamp(const Vec2& v, const Vec2& a, const Vec2& b)
{
    return Vec2(Clamp(v.x, a.x, b.x), Clamp(v.y, a.y, b.y));
}

CUDA_CALLABLE inline Vec3 Clamp(const Vec3& v, const Vec3& a, const Vec3& b)
{
    return Vec3(Clamp(v.x, a.x, b.x), Clamp(v.y, a.y, b.y), Clamp(v.z, a.z, b.z));
}

CUDA_CALLABLE inline Vec4 Clamp(const Vec4& v, const Vec4& a, const Vec4& b)
{
    return Vec4(Clamp(v.x, a.x, b.x), Clamp(v.y, a.y, b.y), Clamp(v.z, a.z, b.z), Clamp(v.w, a.w, b.w));
}


/**
 * @brief Defines a plane via the equation n * p = d
 */
struct Plane
{
    /**
     * @var float n
     * @brief The normal of the plane.
     */

    /**
     * @var float d
     * @brief The offset of the plane.
     */

    CUDA_CALLABLE Plane() : n(Zero), d(0.0f)
    {
    }
    CUDA_CALLABLE Plane(const Vec3& n, float d) : n(n), d(d)
    {
    }
    CUDA_CALLABLE Plane(const Vec3& p0, const Vec3& p1, const Vec3& p2)
    {
        n = (p1 - p0).cross(p2 - p0);
        n.normalize();
        d = p0.dot(n);
    }

    CUDA_CALLABLE float height(const Vec3& p) const
    {
        return p.dot(n) - d;
    }
    CUDA_CALLABLE Vec3 project(const Vec3& p) const
    {
        return p - n * (p.dot(n) - d);
    }

    Vec3 n;
    float d;
};

/**
 * @brief Defines a ray via the equation p = orig + t * dir
 */
struct Ray
{
    /**
     * @var Vec3 orig
     * @brief The origin of the ray.
     */

    /**
     * @var Vec3 dir
     * @brief The direction of the ray.
     */

    CUDA_CALLABLE Ray() : orig(Zero), dir(Zero)
    {
    }
    CUDA_CALLABLE Ray(const Vec3& orig, const Vec3& dir) : orig(orig), dir(dir)
    {
    }

    CUDA_CALLABLE Vec3 at(float t) const
    {
        return orig + t * dir;
    }
    CUDA_CALLABLE bool intersect(const Plane& p, float& t)
    {
        t = dir.dot(p.n);
        if (t == 0.0f)
            return false;
        t = (p.d - orig.dot(p.n)) / t;
        return true;
    }

    Vec3 orig, dir;
};

/**
 * @brief Defines a 2d ray via the equation p = orig + t * dir
 */
struct Ray2
{
    /**
     * @var Vec2 orig
     * @brief The origin of the ray.
     */

    /**
     * @var Vec2 dir
     * @brief The direction of the ray.
     */

    CUDA_CALLABLE Ray2() : orig(Zero), dir(Zero)
    {
    }
    CUDA_CALLABLE Ray2(const Vec2& orig, const Vec2& dir) : orig(orig), dir(dir)
    {
    }

    CUDA_CALLABLE Vec2 at(float t) const
    {
        return orig + t * dir;
    }
    Vec2 orig, dir;
};

//-----------------------------------------------------------------------------

struct Quat;

/**
 * @brief Defines a 3x3 matrix via its columns of type Vec3.
 */

struct Mat33
{
    CUDA_CALLABLE Mat33(){};
    CUDA_CALLABLE Mat33(const Vec3& column0, const Vec3& column1, const Vec3& column2)
        : column0(column0), column1(column1), column2(column2)
    {
    }

    CUDA_CALLABLE Mat33(const Quat& q);
    CUDA_CALLABLE Mat33(_IDENTITY)
        : column0(Vec3(1.0f, 0.0f, 0.0f)), column1(Vec3(0.0f, 1.0f, 0.0f)), column2(Vec3(0.0f, 0.0f, 1.0f)){};

    CUDA_CALLABLE Mat33(_ZERO) : column0(Vec3(Zero)), column1(Vec3(Zero)), column2(Vec3(Zero)){};

    CUDA_CALLABLE Mat33(const Vec3& v)
        : column0(Vec3(v.x, 0.0f, 0.0f)), column1(Vec3(0.0f, v.y, 0.0f)), column2(Vec3(0.0f, 0.0f, v.z)){};

    CUDA_CALLABLE Mat33(float a00, float a01, float a02, float a10, float a11, float a12, float a20, float a21, float a22)
        : column0(Vec3(a00, a10, a20)), column1(Vec3(a01, a11, a21)), column2(Vec3(a02, a12, a22))
    {
    }

    CUDA_CALLABLE bool isZero() const
    {
        return column0.isZero() && column1.isZero() && column2.isZero();
    }

    CUDA_CALLABLE Mat33 getTranspose() const
    {
        Vec3 v0(column0.x, column1.x, column2.x);
        Vec3 v1(column0.y, column1.y, column2.y);
        Vec3 v2(column0.z, column1.z, column2.z);
        return Mat33(v0, v1, v2);
    }

    CUDA_CALLABLE Vec3 operator*(const Vec3& v) const
    {
        return column0 * v.x + column1 * v.y + column2 * v.z;
    }

    CUDA_CALLABLE const Mat33 operator*(float scalar) const
    {
        return Mat33(column0 * scalar, column1 * scalar, column2 * scalar);
    }

    CUDA_CALLABLE Mat33 operator*(const Mat33& M) const
    {
        return Mat33(*this * M.column0, *this * M.column1, *this * M.column2);
    }

    CUDA_CALLABLE Vec3& operator[](unsigned int num)
    {
        return (&column0)[num];
    }
    CUDA_CALLABLE const Vec3& operator[](unsigned int num) const
    {
        return (&column0)[num];
    }

    CUDA_CALLABLE float operator()(unsigned int row, unsigned int col) const
    {
        return (*this)[col][row];
    }

    CUDA_CALLABLE float& operator()(unsigned int row, unsigned int col)
    {
        return (*this)[col][row];
    }

    CUDA_CALLABLE const Mat33 operator+(const Mat33& M) const
    {
        return Mat33(column0 + M.column0, column1 + M.column1, column2 + M.column2);
    }

    CUDA_CALLABLE const Mat33 operator-(const Mat33& M) const
    {
        return Mat33(column0 - M.column0, column1 - M.column1, column2 - M.column2);
    }

    CUDA_CALLABLE Mat33& operator+=(const Mat33& M)
    {
        column0 += M.column0;
        column1 += M.column1;
        column2 += M.column2;
        return *this;
    }

    CUDA_CALLABLE Mat33& operator-=(const Mat33& M)
    {
        column0 -= M.column0;
        column1 -= M.column1;
        column2 -= M.column2;
        return *this;
    }

    CUDA_CALLABLE void operator*=(float scalar)
    {
        column0 *= scalar;
        column1 *= scalar;
        column2 *= scalar;
    }

    CUDA_CALLABLE void operator/=(float scalar)
    {
        column0 /= scalar;
        column1 /= scalar;
        column2 /= scalar;
    }

    CUDA_CALLABLE const Mat33 operator-() const
    {
        return Mat33(-column0, -column1, -column2);
    }

    CUDA_CALLABLE bool operator==(const Mat33& M) const
    {
        return column0 == M.column0 && column1 == M.column1 && column2 == M.column2;
    }

    CUDA_CALLABLE float getDeterminant() const
    {
        return column0.dot(column1.cross(column2));
    }
    CUDA_CALLABLE const Mat33 getInverse() const
    {
        const float det = getDeterminant();
        Mat33 inverse;

        if (det != 0.0)
        {
            const float invDet = 1.0f / det;

            inverse.column0.x = invDet * (column1.y * column2.z - column2.y * column1.z);
            inverse.column0.y = invDet * -(column0.y * column2.z - column2.y * column0.z);
            inverse.column0.z = invDet * (column0.y * column1.z - column0.z * column1.y);

            inverse.column1.x = invDet * -(column1.x * column2.z - column1.z * column2.x);
            inverse.column1.y = invDet * (column0.x * column2.z - column0.z * column2.x);
            inverse.column1.z = invDet * -(column0.x * column1.z - column0.z * column1.x);

            inverse.column2.x = invDet * (column1.x * column2.y - column1.y * column2.x);
            inverse.column2.y = invDet * -(column0.x * column2.y - column0.y * column2.x);
            inverse.column2.z = invDet * (column0.x * column1.y - column1.x * column0.y);

            return inverse;
        }
        else
            return Mat33(Identity);
    }

    CUDA_CALLABLE float dot(const Mat33& M) const
    {
        return column0.dot(M.column0) + column1.dot(M.column1) + column2.dot(M.column2);
    }

    CUDA_CALLABLE void createFrame(const Vec3& axis)
    {
        column0 = axis;
        column0.normalize();
        column1 = column0.perp();
        column2 = column0.cross(column1);
    }

    Vec3 column0, column1, column2;
};

CUDA_CALLABLE inline Mat33 operator*(float f, const Mat33& M)
{
    return Mat33(f * M.column0, f * M.column1, f * M.column2);
}

/**
 * @brief Defines a quaternion with the float components x, y, z and w, where x, y and z define the imaginary part and w
 * the real part.
 */
struct Quat
{
    CUDA_CALLABLE Quat()
    {
    }
    CUDA_CALLABLE Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w)
    {
    }
    CUDA_CALLABLE Quat(_IDENTITY)
    {
        x = y = z = 0.0f;
        w = 1.0f;
    }
    CUDA_CALLABLE Quat(const Vec3& v, float w) : x(v.x), y(v.y), z(v.z), w(w)
    {
    }
    CUDA_CALLABLE Quat(float angleRadians, const Vec3& unitAxis)
    {
        const float a = angleRadians * 0.5f;
        const float s = sinf(a);
        w = cosf(a);
        x = unitAxis.x * s;
        y = unitAxis.y * s;
        z = unitAxis.z * s;
    }
    CUDA_CALLABLE Quat(const Mat33& m)
    {
        if (m.column2.z < 0.0f)
        {
            if (m.column0.x > m.column1.y)
            {
                float t = 1.0f + m.column0.x - m.column1.y - m.column2.z;
                *this = Quat(t, m.column0.y + m.column1.x, m.column2.x + m.column0.z, m.column1.z - m.column2.y) *
                        (0.5f / sqrtf(t));
            }
            else
            {
                float t = 1.0f - m.column0.x + m.column1.y - m.column2.z;
                *this = Quat(m.column0.y + m.column1.x, t, m.column1.z + m.column2.y, m.column2.x - m.column0.z) *
                        (0.5f / sqrtf(t));
            }
        }
        else
        {
            if (m.column0.x < -m.column1.y)
            {
                float t = 1.0f - m.column0.x - m.column1.y + m.column2.z;
                *this = Quat(m.column2.x + m.column0.z, m.column1.z + m.column2.y, t, m.column0.y - m.column1.x) *
                        (0.5f / sqrtf(t));
            }
            else
            {
                float t = 1.0f + m.column0.x + m.column1.y + m.column2.z;
                *this = Quat(m.column1.z - m.column2.y, m.column2.x - m.column0.z, m.column0.y - m.column1.x, t) *
                        (0.5f / sqrtf(t));
            }
        }
    }
    CUDA_CALLABLE Quat(const Vec3& fromAxis, const Vec3& toAxis)
    {
        Vec3 v0 = fromAxis, v1 = toAxis;
        v0.normalize();
        v1.normalize();
        float dot = v0.dot(v1);
        if (dot >= 1.0f)
        {
            x = y = z = 0.0f;
            w = 1.0f;
        }
        else if (dot <= -1.0f)
        {
            Vec3 p = v0.perp();
            x = p.x;
            y = p.y;
            z = p.z;
            w = 0.0f;
        }
        else
        {
            float s = sqrtf((1.0f + dot) * 2.0f);
            float invs = 1.0f / s;
            Vec3 cross = v0.cross(v1);
            x = cross.x * invs;
            y = cross.y * invs;
            z = cross.z * invs;
            w = s * 0.5f;
            normalize();
        }
    }

    CUDA_CALLABLE Vec3 getBasisVector0() const
    {
        const float x2 = x * 2.0f;
        const float w2 = w * 2.0f;
        return Vec3((w * w2) - 1.0f + x * x2, (z * w2) + y * x2, (-y * w2) + z * x2);
    }

    CUDA_CALLABLE Vec3 getBasisVector1() const
    {
        const float y2 = y * 2.0f;
        const float w2 = w * 2.0f;
        return Vec3((-z * w2) + x * y2, (w * w2) - 1.0f + y * y2, (x * w2) + z * y2);
    }

    CUDA_CALLABLE Vec3 getBasisVector2() const
    {
        const float z2 = z * 2.0f;
        const float w2 = w * 2.0f;
        return Vec3((y * w2) + x * z2, (-x * w2) + y * z2, (w * w2) - 1.0f + z * z2);
    }

    CUDA_CALLABLE Quat operator*(float s) const
    {
        return Quat(x * s, y * s, z * s, w * s);
    }

    CUDA_CALLABLE void operator+=(const Quat& q)
    {
        x += q.x;
        y += q.y;
        z += q.z;
        w += q.w;
    }

    CUDA_CALLABLE Quat operator+(const Quat& q)
    {
        return Quat(x + q.x, y + q.y, z + q.z, w + q.w);
    }

    CUDA_CALLABLE Quat operator-(const Quat& q)
    {
        return Quat(x - q.x, y - q.y, z - q.z, w - q.w);
    }

    CUDA_CALLABLE float magnitude()
    {
        return sqrtf(x * x + y * y + z * z + w * w);
    }
    CUDA_CALLABLE void normalize()
    {
        float len = magnitude();
        if (len > 0.0f)
        {
            float invLen = 1.0f / len;
            x *= invLen;
            y *= invLen;
            z *= invLen;
            w *= invLen;
        }
    }

    CUDA_CALLABLE Quat normalized() const
    {
        Quat q = *this;
        q.normalize();
        return q;
    }

    CUDA_CALLABLE Quat getConjugate() const
    {
        return Quat(-x, -y, -z, w);
    }

    CUDA_CALLABLE Quat getInverse() const
    {
        return Quat(-x, -y, -z, w);
    }

    CUDA_CALLABLE const Vec3 rotate(const Vec3& v) const
    {
        const float vx = 2.0f * v.x;
        const float vy = 2.0f * v.y;
        const float vz = 2.0f * v.z;
        const float w2 = w * w - 0.5f;
        const float dot2 = (x * vx + y * vy + z * vz);
        return Vec3((vx * w2 + (y * vz - z * vy) * w + x * dot2), (vy * w2 + (z * vx - x * vz) * w + y * dot2),
                    (vz * w2 + (x * vy - y * vx) * w + z * dot2));
    }

    CUDA_CALLABLE const Vec3 rotateInv(const Vec3& v) const
    {
        const float vx = 2.0f * v.x;
        const float vy = 2.0f * v.y;
        const float vz = 2.0f * v.z;
        const float w2 = w * w - 0.5f;
        const float dot2 = (x * vx + y * vy + z * vz);
        return Vec3((vx * w2 - (y * vz - z * vy) * w + x * dot2), (vy * w2 - (z * vx - x * vz) * w + y * dot2),
                    (vz * w2 - (x * vy - y * vx) * w + z * dot2));
    }

    CUDA_CALLABLE Quat rotate(const Quat& q) const
    {
        return *this * q;
    }

    CUDA_CALLABLE Quat rotateInv(const Quat& q) const
    {
        return getConjugate() * q;
    }

    CUDA_CALLABLE Quat operator*(const Quat& q) const
    {
        return Quat(w * q.x + q.w * x + y * q.z - q.y * z, w * q.y + q.w * y + z * q.x - q.z * x,
                    w * q.z + q.w * z + x * q.y - q.x * y, w * q.w - x * q.x - y * q.y - z * q.z);
    }

    CUDA_CALLABLE Quat operator+(const Quat& q) const
    {
        return Quat(x + q.x, y + q.y, z + q.z, w + q.w);
    }

    CUDA_CALLABLE inline Quat rotateLinear(const Quat& q, const Vec3& omega)
    {
        Quat res = q;
        res += Quat(omega.x, omega.y, omega.z, 0.0) * q * 0.5;
        res.normalize();
        return res;
    }

    CUDA_CALLABLE inline Quat rotateNonlinear(const Quat& q, const Vec3& omega)
    {
        Quat res = q;
        if (!omega.isZero())
        {
            Vec3 n = omega;
            float ang = n.normalize();
            res = Quat(ang, n) * q;
        }
        return res;
    }

    CUDA_CALLABLE Quat lerp(const Quat& v, float s) const
    {
        float t = 1.0f - s;
        Quat q(x * t + v.x * s, y * t + v.y * s, z * t + v.z * s, w * t + v.w * s);
        q.normalize();
        return q;
    }

    CUDA_CALLABLE Quat slerp(const Quat& v, float t) const
    {
        float cosHalfTheta = x * v.x + y * v.y + z * v.z + w * v.w;
        if (fabs(cosHalfTheta) >= 1.0f)
            return *this;

        bool reverse = false;
        if (cosHalfTheta < 0.0f)
        {
            reverse = true;
            cosHalfTheta = -cosHalfTheta;
        }

        float halfTheta = Acos(cosHalfTheta);
        float sinHalfTheta = sqrtf(1.0f - cosHalfTheta * cosHalfTheta);
        float a = 1.0f - t;
        float b = t;

        if (fabs(sinHalfTheta) > 0.001f)
        {
            a = Sin((1.0f - t) * halfTheta) / sinHalfTheta;
            b = Sin(t * halfTheta) / sinHalfTheta;
        }

        if (!reverse)
            return Quat(a * x + b * v.x, a * y + b * v.y, a * z + b * v.z, a * w + b * v.w);
        else
            return Quat(a * x - b * v.x, a * y - b * v.y, a * z - b * v.z, a * w - b * v.w);
    }

    CUDA_CALLABLE Vec3 toEulerAngles() const
    {
        Quat qn = this->normalized();
        Vec3 angles(Zero);

        // First rotation around X
        float sinr_cosp = 2.0f * (qn.w * qn.x + qn.y * qn.z);
        float cosr_cosp = 1.0f - 2.0f * (qn.x * qn.x + qn.y * qn.y);
        angles.x = atan2f(sinr_cosp, cosr_cosp);

        // Second rotation around Y
        float sinp = 2.0f * (qn.w * qn.y - qn.z * qn.x);
        if (fabsf(sinp) >= 1.0f)
            angles.y = copysignf(HalfPi, sinp);
        else
            angles.y = asinf(sinp);

        // Third rotation around Z
        float siny_cosp = 2.0f * (qn.w * qn.z + qn.x * qn.y);
        float cosy_cosp = 1.0f - 2.0f * (qn.y * qn.y + qn.z * qn.z);
        angles.z = atan2f(siny_cosp, cosy_cosp);
        return angles;
    }

    CUDA_CALLABLE static Quat fromEulerAngles(const Vec3& angles)
    {
        Quat q;

        float cy = cosf(angles.x * 0.5f);
        float sy = sinf(angles.x * 0.5f);
        float cp = cosf(angles.y * 0.5f);
        float sp = sinf(angles.y * 0.5f);
        float cr = cosf(angles.z * 0.5f);
        float sr = sinf(angles.z * 0.5f);

        q.w = cr * cp * cy + sr * sp * sy;
        q.z = sr * cp * cy - cr * sp * sy;
        q.y = cr * sp * cy + sr * cp * sy;
        q.x = cr * cp * sy - sr * sp * cy;

        return q;
    }

    float x, y, z, w;
};


CUDA_CALLABLE inline Quat operator*(float f, const Quat& q)
{
    return Quat(f * q.x, f * q.y, f * q.z, f * q.w);
}


CUDA_CALLABLE inline Mat33::Mat33(const Quat& q)
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    float yz = y2 * z, yw = y2 * w, zw = z2 * w;
    column0 = Vec3(1.0f - yy - zz, xy + zw, xz - yw);
    column1 = Vec3(xy - zw, 1.0f - xx - zz, yz + xw);
    column2 = Vec3(xz + yw, yz - xw, 1.0f - xx - yy);
}

/**
 * @brief Defines a rigid transformation via a translation and a rotation.
 */
struct Transform
{
    /**
     * @var Vec3 p
     * @brief The translation of the transform.
     */

    /**
     * @brief Unit quaternion q
     * @brief The rotation of the transform.
     */

public:
    Vec3 p;
    Quat q;

    CUDA_CALLABLE Transform(){};
    CUDA_CALLABLE Transform(_IDENTITY) : p(Zero), q(Identity){};
    CUDA_CALLABLE Transform(const Vec3& p, const Quat& q) : p(p), q(q){};
    CUDA_CALLABLE Transform(const Vec3& p) : p(p), q(Identity){};
    CUDA_CALLABLE Transform(const Quat& q) : p(Zero), q(q){};
    CUDA_CALLABLE Transform getInverse() const
    {
        return Transform(q.rotateInv(p * -1.0f), q.getConjugate());
    }

    CUDA_CALLABLE Transform transform(const Transform& t) const
    {
        return Transform(q.rotate(t.p) + p, q * t.q);
    }

    CUDA_CALLABLE Transform transformInv(const Transform& t) const
    {
        Quat qinv = q.getConjugate();
        return Transform(qinv.rotate(t.p - p), qinv * t.q);
    }

    CUDA_CALLABLE Vec3 transform(const Vec3& v) const
    {
        return q.rotate(v) + p;
    }

    CUDA_CALLABLE Vec3 transformInv(const Vec3& v) const
    {
        return q.rotateInv(v - p);
    }

    CUDA_CALLABLE Vec3 rotate(const Vec3& v) const
    {
        return q.rotate(v);
    }

    CUDA_CALLABLE Vec3 rotateInv(const Vec3& v) const
    {
        return q.rotateInv(v);
    }

    CUDA_CALLABLE Plane transform(const Plane& in) const
    {
        Plane out;
        out.n = q.rotate(in.n);
        out.d = in.d + out.n.dot(p);
        return out;
    }

    CUDA_CALLABLE Ray transform(const Ray& r) const
    {
        return Ray(transform(r.orig), rotate(r.dir));
    }

    CUDA_CALLABLE Ray transformInv(const Ray& r) const
    {
        return Ray(transformInv(r.orig), rotateInv(r.dir));
    }

    CUDA_CALLABLE Plane transformInv(const Plane& in) const
    {
        Plane out;
        out.n = q.rotateInv(in.n);
        out.d = in.d - out.n.dot(q.rotateInv(p));
        return out;
    }

    CUDA_CALLABLE Transform operator*(const Transform& t) const
    {
        return transform(t);
    }
};

struct Mat44;

/**
 * @brief Defines an affine transformation.
 * An affine transformation is a transformation that preserves straight lines and planes.
 * It can be represented as a 3x3 matrix A and a translation vector b
 * or as a 4x4 matrix with the last row being (0, 0, 0, 1).
 */

struct AffineTransform
{
    /**
     * @var Mat33 A
     * @brief The linear transformation of the affine transform.
     */

    /**
     * @var Vec3 b
     * @brief The translation of the affine transform.
     */
public:
    Mat33 A;
    Vec3 b;

    CUDA_CALLABLE AffineTransform(){};

    /**
     * @brief Defines an affine transformation which maps all points to the origin.
     */
    CUDA_CALLABLE AffineTransform(_ZERO) : A(Zero), b(Zero){};

    /**
     * @brief Defines an affine transformation which maps all points to themselves.
     */
    CUDA_CALLABLE AffineTransform(_IDENTITY) : A(Identity), b(Zero){};

    /**
     * @brief Defines an affine transformation with the given linear transformation and translation.
     */
    CUDA_CALLABLE AffineTransform(const Mat33& A, const Vec3& b) : A(A), b(b){};

    /**
     * @brief Defines an affine transformation with the given rigid transform.
     */
    CUDA_CALLABLE AffineTransform(const Transform& t, const Vec3& scale = Vec3(1.0f, 1.0f, 1.0f))
        : A(Mat33(t.q) * Mat33(scale)), b(t.p){};

    /**
     * @brief Defines an affine transformation with the given 4x4 matrix.
     */
    CUDA_CALLABLE AffineTransform(const Mat44 m);

    /**
     * @brief Tests whether the affine transformation is the zero transformation.
     */
    CUDA_CALLABLE bool isZero() const
    {
        return A.isZero() && b.isZero();
    }

    /**
     * @brief Transforms the given vector.
     */
    CUDA_CALLABLE Vec3 transform(const Vec3& v) const
    {
        return A * v + b;
    }

    /**
     * @brief Transforms the given ray.
     */
    CUDA_CALLABLE Ray transform(const Ray& r) const
    {
        return Ray(transform(r.orig), A * r.dir);
    }

    CUDA_CALLABLE AffineTransform transform(const AffineTransform& t) const
    {
        return AffineTransform(A * t.A, A * t.b + b);
    }

    /**
     * @brief Get the matrix M that can be used to transform normals such that n' = M n
     */
    CUDA_CALLABLE Mat33 getNormalMatrix() const
    {
        return A.getInverse().getTranspose();
    }

    /**
     * @brief Get the inverse of the affine transformation.
     */
    CUDA_CALLABLE AffineTransform getInverse() const
    {
        Mat33 Ainv = A.getInverse();
        return AffineTransform(Ainv, Ainv * -b);
    }

    /**
     * @brief Multiplies the affine transformation with a scalar.
     */
    CUDA_CALLABLE const AffineTransform operator*(float scalar) const
    {
        return AffineTransform(A * scalar, b * scalar);
    }

    /**
     * @brief Adds a given transform.
     */
    CUDA_CALLABLE void operator+=(const AffineTransform& at)
    {
        A += at.A;
        b += at.b;
    }

    /**
     * @brief Subtracts a given transform.
     */
    CUDA_CALLABLE void operator-=(const AffineTransform& at)
    {
        A -= at.A;
        b -= at.b;
    }

    CUDA_CALLABLE AffineTransform operator*(const AffineTransform& t) const
    {
        return transform(t);
    }
};

/**
 * @brief Defines a 4x4 matrix via its columns of type Vec4.
 */
struct Mat44
{
    CUDA_CALLABLE Mat44()
    {
    }

    CUDA_CALLABLE Mat44(const Vec4& column0, const Vec4& column1, const Vec4& column2, const Vec4& column3)
        : column0(column0), column1(column1), column2(column2), column3(column3)
    {
    }

    CUDA_CALLABLE Mat44(_IDENTITY)
        : column0(Vec4(1.0, 0.0, 0.0, 0.0)),
          column1(Vec4(0.0, 1.0, 0.0, 0.0)),
          column2(Vec4(0.0, 0.0, 1.0, 0.0)),
          column3(Vec4(0.0, 0.0, 0.0, 1.0)){};

    CUDA_CALLABLE Mat44(_ZERO) : column0(Vec4(Zero)), column1(Vec4(Zero)), column2(Vec4(Zero)), column3(Vec4(Zero)){};

    CUDA_CALLABLE Mat44(float m[])
        : column0(m[0], m[1], m[2], m[3]),
          column1(m[4], m[5], m[6], m[7]),
          column2(m[8], m[9], m[10], m[11]),
          column3(m[12], m[13], m[14], m[15])
    {
    }

    CUDA_CALLABLE Mat44(float a00,
                        float a01,
                        float a02,
                        float a03,
                        float a10,
                        float a11,
                        float a12,
                        float a13,
                        float a20,
                        float a21,
                        float a22,
                        float a23,
                        float a30,
                        float a31,
                        float a32,
                        float a33)
        : column0(a00, a10, a20, a30),
          column1(a01, a11, a21, a31),
          column2(a02, a12, a22, a32),
          column3(a03, a13, a23, a33)
    {
    }

    CUDA_CALLABLE Mat44(const Transform& trans, const Vec3& scale = Vec3(1.0f, 1.0f, 1.0f))
        : column0(trans.q.getBasisVector0() * scale.x),
          column1(trans.q.getBasisVector1() * scale.y),
          column2(trans.q.getBasisVector2() * scale.z),
          column3(trans.p, 1.0f)
    {
    }

    CUDA_CALLABLE Mat44(const Vec3& col0, const Vec3& col1, const Vec3& col2, const Vec3& col3)
        : column0(col0, 0), column1(col1, 0), column2(col2, 0), column3(col3, 1.0f)
    {
    }

    CUDA_CALLABLE Mat44(const AffineTransform& t)
        : column0(t.A.column0), column1(t.A.column1), column2(t.A.column2), column3(Vec4(t.b, 1.0f))
    {
    }

    CUDA_CALLABLE void getColumnMajor(float* f) const
    {
        f[0] = column0.x;
        f[4] = column1.x;
        f[8] = column2.x;
        f[12] = column3.x;
        f[1] = column0.y;
        f[5] = column1.y;
        f[9] = column2.y;
        f[13] = column3.y;
        f[2] = column0.z;
        f[6] = column1.z;
        f[10] = column2.z;
        f[14] = column3.z;
        f[3] = column0.w;
        f[7] = column1.w;
        f[11] = column2.w;
        f[15] = column3.w;
    }

    CUDA_CALLABLE Mat44 getTranspose() const
    {
        Vec4 v0(column0.x, column1.x, column2.x, column3.x);
        Vec4 v1(column0.y, column1.y, column2.y, column3.y);
        Vec4 v2(column0.z, column1.z, column2.z, column3.z);
        Vec4 v3(column0.w, column1.w, column2.w, column3.w);
        return Mat44(v0, v1, v2, v3);
    }

    CUDA_CALLABLE const Mat44 getInverse() const
    {
        float inv[16], m[16];
        getColumnMajor(m);

        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
                 m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
                 m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
                 m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
                  m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
                 m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
                 m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
                 m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
                  m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
                 m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
                 m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
                  m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
                  m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
                 m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
                 m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
                  m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
                  m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

        float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

        if (det == 0)
            return Mat44(Zero);

        det = 1.0f / det;

        for (int i = 0; i < 16; i++)
            inv[i] *= det;

        return Mat44(inv);
    }

    CUDA_CALLABLE Vec4 transform(const Vec4& v) const
    {
        return column0 * v.x + column1 * v.y + column2 * v.z + column3 * v.w;
    }

    CUDA_CALLABLE Mat33 get33() const
    {
        return Mat33(column0.getXYZ(), column1.getXYZ(), column2.getXYZ());
    }

    CUDA_CALLABLE Vec3 getP() const
    {
        return column3.getXYZ();
    }

    CUDA_CALLABLE Transform getRigidTransform() const
    {
        return Transform(column3.getXYZ(), Quat(get33()));
    }

    CUDA_CALLABLE Vec4 operator*(const Vec4& v) const
    {
        return transform(v);
    }

    CUDA_CALLABLE Mat44 operator*(float scalar) const
    {
        return Mat44(column0 * scalar, column1 * scalar, column2 * scalar, column3 * scalar);
    }

    CUDA_CALLABLE Mat44 operator*(const Mat44& other) const
    {
        return Mat44(
            transform(other.column0), transform(other.column1), transform(other.column2), transform(other.column3));
    }

    CUDA_CALLABLE Mat44 operator+(const Mat44& M) const
    {
        return Mat44(column0 + M.column0, column1 + M.column1, column2 + M.column2, column3 + M.column3);
    }

    CUDA_CALLABLE Mat44 operator-(const Mat44& M) const
    {
        return Mat44(column0 - M.column0, column1 - M.column1, column2 - M.column2, column3 - M.column3);
    }

    CUDA_CALLABLE Mat44& operator+=(const Mat44& M)
    {
        column0 += M.column0;
        column1 += M.column1;
        column2 += M.column2;
        column3 += M.column3;
        return *this;
    }

    CUDA_CALLABLE Mat44& operator-=(const Mat44& M)
    {
        column0 -= M.column0;
        column1 -= M.column1;
        column2 -= M.column2;
        column3 -= M.column3;
        return *this;
    }

    CUDA_CALLABLE bool operator==(const Mat44& M) const
    {
        return column0 == M.column0 && column1 == M.column1 && column2 == M.column2 && column3 == M.column3;
    }

    CUDA_CALLABLE Vec3 transform(const Vec3& v) const
    {
        return transform(Vec4(v, 1.0)).getXYZ();
    }

    CUDA_CALLABLE Vec3 rotate(const Vec3& v) const
    {
        return transform(Vec4(v, 0.0)).getXYZ();
    }

    CUDA_CALLABLE Vec4& getColumn(int nr)
    {
        return *(&column0 + nr);
    }
    CUDA_CALLABLE const Vec4& getColumn(int nr) const
    {
        return *(&column0 + nr);
    }

    Vec4 column0, column1, column2, column3;
};

CUDA_CALLABLE inline AffineTransform::AffineTransform(const Mat44 m)
    : A(m.column0.getXYZ(), m.column1.getXYZ(), m.column2.getXYZ()), b(m.column3.getXYZ()){};

CUDA_CALLABLE inline Mat44 operator*(float f, const Mat44& M)
{
    return Mat44(f * M.column0, f * M.column1, f * M.column2, f * M.column3);
}

/**
 * @brief Defines a two dimensional bounding box.
 */
struct Bounds2
{
    /**
     * @var Vec2 minimum
     * @brief the minimum point of the bounding box.
     */
    /**
     * @var Vec2 maximum
     * @brief the maximum point of the bounding box.
     */

    CUDA_CALLABLE Bounds2()
    {
    }
    CUDA_CALLABLE Bounds2(const Vec2& minimum, const Vec2& maximum) : minimum(minimum), maximum(maximum)
    {
    }
    CUDA_CALLABLE Bounds2(const Vec2& p) : minimum(p), maximum(p)
    {
    }
    CUDA_CALLABLE Bounds2(_EMPTY)
    {
        setEmpty();
    }
    CUDA_CALLABLE Bounds2(_ZERO) : minimum(Vec2(Zero)), maximum(Vec2(Zero))
    {
    }

    CUDA_CALLABLE void setEmpty()
    {
        minimum = Vec2(MaxFloat, MaxFloat);
        maximum = Vec2(-MaxFloat, -MaxFloat);
    }
    CUDA_CALLABLE bool isEmpty()
    {
        return minimum.x > maximum.x;
    }

    CUDA_CALLABLE void include(const Vec2& p)
    {
        minimum.setMin(p);
        maximum.setMax(p);
    }

    CUDA_CALLABLE void include(const Bounds2& b)
    {
        minimum.setMin(b.minimum);
        maximum.setMax(b.maximum);
    }

    CUDA_CALLABLE void expand(float f)
    {
        minimum -= Vec2(f, f);
        maximum += Vec2(f, f);
    }

    CUDA_CALLABLE bool intersect(const Bounds2& b) const
    {
        return !(b.minimum.x > maximum.x || minimum.x > b.maximum.x || b.minimum.y > maximum.y ||
                 minimum.y > b.maximum.y);
    }

    CUDA_CALLABLE bool contain(const Vec2& v) const
    {
        return !(v.x < minimum.x || v.x > maximum.x || v.y < minimum.y || v.y > maximum.y);
    }

    CUDA_CALLABLE Vec2 getDimensions() const
    {
        return maximum - minimum;
    }

    CUDA_CALLABLE Vec2 getHalfExtents() const
    {
        return getDimensions() * 0.5f;
    }

    CUDA_CALLABLE Vec2 getCenter() const
    {
        return (maximum + minimum) * 0.5f;
    }

    Vec2 minimum, maximum;
};


/**
 * @brief Defines a three dimensional bounding box.
 */
struct Bounds3
{
    /**
     * @var Vec3 minimum
     * @brief the minimum point of the bounding box.
     */
    /**
     * @var Vec3 maximum
     * @brief the maximum point of the bounding box.
     */

    CUDA_CALLABLE Bounds3()
    {
    }
    CUDA_CALLABLE Bounds3(const Vec3& minimum, const Vec3& maximum) : minimum(minimum), maximum(maximum)
    {
    }
    CUDA_CALLABLE Bounds3(const Vec3& p) : minimum(p), maximum(p)
    {
    }
    CUDA_CALLABLE Bounds3(_EMPTY)
    {
        setEmpty();
    }
    CUDA_CALLABLE Bounds3(_ZERO) : minimum(Vec3(Zero)), maximum(Vec3(Zero))
    {
    }
    CUDA_CALLABLE Bounds3(const Transform& pose, const Vec3& halfExtents)
    {
        Vec3 c0 = pose.q.getBasisVector0() * halfExtents.x;
        Vec3 c1 = pose.q.getBasisVector1() * halfExtents.y;
        Vec3 c2 = pose.q.getBasisVector2() * halfExtents.z;
        Vec3 w;
        w.x = fabsf(c0.x) + fabsf(c1.x) + fabsf(c2.x);
        w.y = fabsf(c0.y) + fabsf(c1.y) + fabsf(c2.y);
        w.z = fabsf(c0.z) + fabsf(c1.z) + fabsf(c2.z);
        minimum = pose.p - w;
        maximum = pose.p + w;
    }
    CUDA_CALLABLE Bounds3(const AffineTransform& pose, const Vec3& halfExtents)
    {
        Vec3 c0 = pose.A.column0 * halfExtents.x;
        Vec3 c1 = pose.A.column1 * halfExtents.y;
        Vec3 c2 = pose.A.column2 * halfExtents.z;
        Vec3 w;
        w.x = fabsf(c0.x) + fabsf(c1.x) + fabsf(c2.x);
        w.y = fabsf(c0.y) + fabsf(c1.y) + fabsf(c2.y);
        w.z = fabsf(c0.z) + fabsf(c1.z) + fabsf(c2.z);
        minimum = pose.b - w;
        maximum = pose.b + w;
    }
    CUDA_CALLABLE void setEmpty()
    {
        minimum = Vec3(MaxFloat, MaxFloat, MaxFloat);
        maximum = Vec3(-MaxFloat, -MaxFloat, -MaxFloat);
    }
    CUDA_CALLABLE bool isEmpty()
    {
        return minimum.x > maximum.x;
    }

    CUDA_CALLABLE void include(const Vec3& p)
    {
        minimum.setMin(p);
        maximum.setMax(p);
    }

    CUDA_CALLABLE void include(const Bounds3& b)
    {
        minimum.setMin(b.minimum);
        maximum.setMax(b.maximum);
    }

    CUDA_CALLABLE void expand(float f)
    {
        minimum -= Vec3(f, f, f);
        maximum += Vec3(f, f, f);
    }

    CUDA_CALLABLE bool intersect(const Bounds3& b) const
    {
        return !(b.minimum.x > maximum.x || minimum.x > b.maximum.x || b.minimum.y > maximum.y ||
                 minimum.y > b.maximum.y || b.minimum.z > maximum.z || minimum.z > b.maximum.z);
    }

    CUDA_CALLABLE void getIntersection(const Bounds3& b0, const Bounds3& b1)
    {
        if (!b0.intersect(b1))
        {
            setEmpty();
        }
        else
        {
            minimum = b0.minimum;
            minimum.setMax(b1.minimum);
            maximum = b0.maximum;
            maximum.setMin(b1.maximum);
        }
    }

    CUDA_CALLABLE bool contain(const Vec3& v) const
    {
        return !(v.x < minimum.x || v.x > maximum.x || v.y < minimum.y || v.y > maximum.y || v.z < minimum.z ||
                 v.z > maximum.z);
    }

    CUDA_CALLABLE bool contain(const Bounds3& b) const
    {
        return contain(b.minimum) && contain(b.maximum);
    }

    CUDA_CALLABLE Vec3 getDimensions() const
    {
        return maximum - minimum;
    }

    CUDA_CALLABLE Vec3 getHalfExtents() const
    {
        return getDimensions() * 0.5f;
    }

    CUDA_CALLABLE Vec3 getCenter() const
    {
        return (maximum + minimum) * 0.5f;
    }

    CUDA_CALLABLE Bounds3 transform(const Transform& trans) const
    {
        return Bounds3(Transform(trans.transform(getCenter()), trans.q), getHalfExtents());
    }

    CUDA_CALLABLE Bounds3 transformInv(const Transform& trans) const
    {
        return transform(trans.getInverse());
    }

    CUDA_CALLABLE Bounds3 transform(const AffineTransform& trans) const
    {
        return Bounds3(AffineTransform(trans.A, trans.transform(getCenter())), getHalfExtents());
    }

    Vec3 minimum, maximum;
};


/**
 * @brief Defines a three dimensional oriented bounding box.
 */
struct OBB
{
    /**
     * @var Transform trans
     * @brief The rigid transformation of the zero centered bounding box.
     */
    /**
     * @var Vec3 halfExtents
     * @brief The half edge lengths of the bounding box.
     */

    CUDA_CALLABLE OBB()
    {
    }
    CUDA_CALLABLE OBB(const Transform& trans, const Vec3& halfExtents) : trans(trans), halfExtents(halfExtents)
    {
    }
    CUDA_CALLABLE OBB(_EMPTY)
    {
        setEmpty();
    }

    CUDA_CALLABLE void setEmpty()
    {
        trans = Transform(Identity);
        halfExtents = Vec3(Zero);
    }

    CUDA_CALLABLE bool isEmpty()
    {
        return halfExtents.isZero();
    }

    bool contains(const Vec3& p) const
    {
        Vec3 localP = trans.transformInv(p);
        return fabsf(localP.x) <= halfExtents.x && fabsf(localP.y) <= halfExtents.y && fabsf(localP.z) <= halfExtents.z;
    }

    Transform trans;
    Vec3 halfExtents;
};

/**
 * @brief Defines a two dimensional oriented bounding box.
 */
struct OBB2
{
    /**
     * @var Transform trans
     * @brief The rigid transformation of the zero centered bounding box.
     */
    /**
     * @var Vec3 halfExtents
     * @brief The half edge lengths of the bounding box.
     */

    CUDA_CALLABLE OBB2()
    {
    }
    CUDA_CALLABLE OBB2(const Vec2& center, const Vec2& axis, const Vec2& halfExtents)
        : center(center), axis(axis), halfExtents(halfExtents)
    {
    }
    CUDA_CALLABLE OBB2(_EMPTY)
    {
        setEmpty();
    }

    CUDA_CALLABLE void setEmpty()
    {
        center = Vec2(Zero);
        axis = Vec2(Zero);
        halfExtents = Vec2(Zero);
    }

    CUDA_CALLABLE bool isEmpty()
    {
        return halfExtents.isZero();
    }

    bool contains(const Vec2& p) const
    {
        float x = (p - center).dot(axis);
        float y = (p - center).cross(axis.perp());
        return fabsf(x) <= halfExtents.x && fabsf(y) <= halfExtents.y;
    }

    Vec2 center;
    Vec2 axis;
    Vec2 halfExtents;
};

// --------------------------------------------------------------------------
/**
 * @brief Computes the outer product v0 v1^T of two vectors which yields a matrix.
 */
inline CUDA_CALLABLE Mat33 outerProduct(const Vec3& v0, const Vec3& v1)
{
    Mat33 M;
    M.column0 = v0 * v1.x;
    M.column1 = v0 * v1.y;
    M.column2 = v0 * v1.z;
    return M;
}

/**
 * @brief Defines a quadric error measure used for mesh simplification.
 */
struct Quadric
{
public:
    CUDA_CALLABLE void zero()
    {
        a00 = 0.0f;
        a01 = 0.0f;
        a02 = 0.0f;
        a03 = 0.0f;
        a11 = 0.0f;
        a12 = 0.0f;
        a13 = 0.0f;
        a22 = 0.0f;
        a23 = 0.0f;
        a33 = 0.0f;
    }

    // generate quadric from plane
    CUDA_CALLABLE void setFromPlane(const Vec3& v0, const Vec3& v1, const Vec3& v2)
    {
        Vec3 n = (v1 - v0).cross(v2 - v0);
        n.normalize();
        float d = -n.dot(v0);
        a00 = n.x * n.x;
        a01 = n.x * n.y;
        a02 = n.x * n.z;
        a03 = n.x * d;
        a11 = n.y * n.y;
        a12 = n.y * n.z;
        a13 = n.y * d;
        a22 = n.z * n.z;
        a23 = n.z * d;
        a33 = d * d;
    }

    CUDA_CALLABLE Quadric operator+(const Quadric& q) const
    {
        Quadric sum;
        sum.a00 = a00 + q.a00;
        sum.a01 = a01 + q.a01;
        sum.a02 = a02 + q.a02;
        sum.a03 = a03 + q.a03;
        sum.a11 = a11 + q.a11;
        sum.a12 = a12 + q.a12;
        sum.a13 = a13 + q.a13;
        sum.a22 = a22 + q.a22;
        sum.a23 = a23 + q.a23;
        sum.a33 = a33 + q.a33;
        return sum;
    }

    CUDA_CALLABLE Quadric operator-(const Quadric& q) const
    {
        Quadric sum;
        sum.a00 = a00 - q.a00;
        sum.a01 = a01 - q.a01;
        sum.a02 = a02 - q.a02;
        sum.a03 = a03 - q.a03;
        sum.a11 = a11 - q.a11;
        sum.a12 = a12 - q.a12;
        sum.a13 = a13 - q.a13;
        sum.a22 = a22 - q.a22;
        sum.a23 = a23 - q.a23;
        sum.a33 = a33 - q.a33;
        return sum;
    }

    CUDA_CALLABLE void operator+=(const Quadric& q)
    {
        a00 += q.a00;
        a01 += q.a01;
        a02 += q.a02;
        a03 += q.a03;
        a11 += q.a11;
        a12 += q.a12;
        a13 += q.a13;
        a22 += q.a22;
        a23 += q.a23;
        a33 += q.a33;
    }

    CUDA_CALLABLE float outerProduct(const Vec3& v)
    {
        double vx = v.x;
        double vy = v.y;
        double vz = v.z;
        double res = a00 * vx * vx + 2.0 * a01 * vx * vy + 2.0 * a02 * vx * vz + 2.0 * a03 * vx + a11 * vy * vy +
                     2.0 * a12 * vy * vz + 2.0 * a13 * vy + a22 * vz * vz + 2.0 * a23 * vz + a33;
        return (float)res;
    }

    CUDA_CALLABLE float computeMinimum(const Ray& ray)
    {
        double ax = ray.orig.x;
        double ay = ray.orig.y;
        double az = ray.orig.z;

        double bx = ray.dir.x;
        double by = ray.dir.y;
        double bz = ray.dir.z;

        double a = a00 * bx * bx + a11 * by * by + a22 * bz * bz + 2.0 * a01 * bx * by + 2.0 * a02 * bx * bz +
                   2.0 * a12 * by * bz;

        double b = a00 * 2.0 * ax * bx + a11 * 2.0 * ay * by + a22 * 2.0 * az * bz + 2.0 * a01 * (ax * by + ay * bx) +
                   2.0 * a02 * (ax * bz + az * bx) + 2.0 * a12 * (ay * bz + az * by) + 2.0 * a03 * bx + 2.0 * a13 * by +
                   2.0 * a23 * bz;

        return a != 0.0 ? (float)(-b / (2.0 * a)) : 0.0f;
    }

    float a00, a01, a02, a03;
    float a11, a12, a13;
    float a22, a23;
    float a33;
};

/**
 * @brief Defines a vector of general length
 */
struct Vec
{
    Vec(size_t size = 0)
    {
        data.resize(size, 0.0f);
    }

    Vec(const Vec& v)
    {
        data.resize(v.data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = v.data[i];
        }
    }

    Vec& operator=(const Vec& v)
    {
        if (this != &v) // Self-assignment check
        {
            data.resize(v.data.size());
            for (size_t i = 0; i < data.size(); ++i)
            {
                data[i] = v.data[i];
            }
        }
        return *this;
    }

    size_t size() const
    {
        return data.size();
    }

    void resize(size_t size)
    {
        data.resize(size, 0.0f);
    }

    void setZero()
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = 0.0f;
        }
    }

    float& operator[](unsigned int index)
    {
        return data[index];
    }

    const float& operator[](unsigned int index) const
    {
        return data[index];
    }

    void get(Vec2& v) const
    {
        v.x = data.size() > 0 ? data[0] : 0.0f;
        v.y = data.size() > 1 ? data[1] : 0.0f;
    }

    void get(Vec3& v) const
    {
        v.x = data.size() > 0 ? data[0] : 0.0f;
        v.y = data.size() > 1 ? data[1] : 0.0f;
        v.z = data.size() > 2 ? data[2] : 0.0f;
    }

    void get(Vec4& v) const
    {
        v.x = data.size() > 0 ? data[0] : 0.0f;
        v.y = data.size() > 1 ? data[1] : 0.0f;
        v.z = data.size() > 2 ? data[2] : 0.0f;
        v.w = data.size() > 3 ? data[3] : 0.0f;
    }

    Vec2 getVec2() const
    {
        Vec2 v;
        get(v);
        return v;
    }

    Vec3 getVec3() const
    {
        Vec3 v;
        get(v);
        return v;
    }

    Vec4 getVec4() const
    {
        Vec4 v;
        get(v);
        return v;
    }

    void set(const Vec2& v)
    {
        data.resize(2);
        data[0] = v.x;
        data[1] = v.y;
    }

    void set(const Vec3& v)
    {
        data.resize(3);
        data[0] = v.x;
        data[1] = v.y;
        data[2] = v.z;
    }

    void set(const Vec4& v)
    {
        data.resize(4);
        data[0] = v.x;
        data[1] = v.y;
        data[2] = v.z;
        data[3] = v.w;
    }

    bool operator==(const Vec& v) const
    {
        if (data.size() != v.data.size())
        {
            return false;
        }
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i] != v.data[i])
                return false;
        }
        return true;
    }

    bool operator!=(const Vec& v) const
    {
        return !(*this == v);
    }

    bool isZero() const
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i] != 0.0f)
                return false;
        }
        return true;
    }

    Vec operator-() const
    {
        Vec v(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            v.data[i] = -data[i];
        }
        return v;
    }

    Vec operator+(const Vec& v) const
    {
        size_t newSize = Max(data.size(), v.data.size());
        Vec r(newSize);
        for (size_t i = 0; i < newSize; ++i)
        {
            float a = i < data.size() ? data[i] : 0.0f;
            float b = i < v.data.size() ? v.data[i] : 0.0f;
            r.data[i] = a + b;
        }
        return r;
    }

    Vec operator-(const Vec& v) const
    {
        size_t newSize = Max(data.size(), v.data.size());
        Vec r(newSize);
        for (size_t i = 0; i < newSize; ++i)
        {
            float a = i < data.size() ? data[i] : 0.0f;
            float b = i < v.data.size() ? v.data[i] : 0.0f;
            r.data[i] = a - b;
        }
        return r;
    }

    void operator+=(const Vec& v)
    {
        for (size_t i = 0; i < Min(data.size(), v.data.size()); ++i)
        {
            data[i] += v.data[i];
        }
    }

    void operator-=(const Vec& v)
    {
        for (size_t i = 0; i < Min(data.size(), v.data.size()); ++i)
        {
            data[i] -= v.data[i];
        }
    }

    void operator*=(float f)
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] *= f;
        }
    }

    void operator/=(float f)
    {
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] /= f;
        }
    }

    Vec operator*(float f) const
    {
        Vec r(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            r.data[i] = data[i] * f;
        }
        return r;
    }

    Vec operator/(float f) const
    {
        Vec r(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            r.data[i] = data[i] / f;
        }
        return r;
    }

    float length() const
    {
        return sqrtf(lengthSquared());
    }

    float lengthSquared() const
    {
        float l = 0.0f;
        for (size_t i = 0; i < data.size(); ++i)
        {
            l += data[i] * data[i];
        }
        return l;
    }

    float normalize()
    {
        float l = length();
        if (l > 0.0f)
        {
            *this /= l;
        }
        return l;
    }

    std::vector<float> data;
};


/**
 * @brief Adds an integer value to an int variable. The operation is atomic if the code is executed on the GPU.
 * @param var The variable to which the value is added.
 * @param val The value to add.
 * @return The value of the variable before the addition.
 */
CUDA_CALLABLE_DEVICE inline int AtomicAdd(int* var, int val)
{
#ifdef __CUDACC__
    return atomicAdd(var, val);
#else
    *var += val;
    return *var - val;
#endif
}

/**
 * @brief Adds a float value to an float variable. The operation is atomic if the code is executed on the GPU.
 * @param var The variable to which the value is added.
 * @param val The value to add.
 * @return The value of the variable before the addition.
 */
CUDA_CALLABLE_DEVICE inline float AtomicAdd(float* var, float val)
{
#ifdef __CUDACC__
    return atomicAdd(var, val);
#else
    *var += val;
    return *var - val;
#endif
}

/**
 * @brief Adds a Vec3 value to an Vec3 variable. The operation is atomic if the code is executed on the GPU.
 * @param var The variable to which the value is added.
 * @param val The value to add.
 * @return The value of the variable before the addition.
 */
CUDA_CALLABLE_DEVICE inline Vec3 AtomicAdd(Vec3* v, Vec3 val)
{
    Vec3 old;
#ifdef __CUDACC__
    old.x = atomicAdd(&v->x, val.x);
    old.y = atomicAdd(&v->y, val.y);
    old.z = atomicAdd(&v->z, val.z);
    return old;
#else
    *v += val;
    return *v - val;
#endif
}

/**
 * @brief Adds a Quat value to an Quat variable. The operation is atomic if the code is executed on the GPU.
 * @param var The variable to which the value is added.
 * @param val The value to add.
 * @return The value of the variable before the addition.
 */
CUDA_CALLABLE_DEVICE inline Quat AtomicAdd(Quat* q, Quat val)
{
    Quat old;
#ifdef __CUDACC__
    old.x = atomicAdd(&q->x, val.x);
    old.y = atomicAdd(&q->y, val.y);
    old.z = atomicAdd(&q->z, val.z);
    old.w = atomicAdd(&q->w, val.w);
    return old;
#else
    *q += val;
    return *q - val;
#endif
}

/**
 * @brief Adds a Quadric to a Quadric variable. The operation is atomic if the code is executed on the GPU.
 * @param var The variable to which the value is added.
 * @param val The value to add.
 * @return The value of the variable before the addition.
 */
CUDA_CALLABLE_DEVICE inline Quadric AtomicAdd(Quadric* q, Quadric val)
{
    Quadric old;
#ifdef __CUDACC__
    old.a00 = atomicAdd(&q->a00, val.a00);
    old.a01 = atomicAdd(&q->a01, val.a01);
    old.a02 = atomicAdd(&q->a02, val.a02);
    old.a03 = atomicAdd(&q->a03, val.a03);
    old.a11 = atomicAdd(&q->a11, val.a11);
    old.a12 = atomicAdd(&q->a12, val.a12);
    old.a13 = atomicAdd(&q->a13, val.a13);
    old.a22 = atomicAdd(&q->a22, val.a22);
    old.a23 = atomicAdd(&q->a23, val.a23);
    old.a33 = atomicAdd(&q->a33, val.a33);
    return old;
#else
    *q += val;
    old = *q - val;
#endif
    return old;
}

