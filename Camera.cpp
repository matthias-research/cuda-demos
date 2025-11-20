// by Matthias Müller

#include "Camera.h"

#define DEGS_PER_PIXEL 0.5f

// -----------------------------------------------------------------------------------------------------------------
void Camera::init()
{
    up = Vec3(0.0f, 1.0f, 0.0f);
    speed = 0.1f;
    fov = 40.0f;
    resetView();
    for (int i = 0; i < 16; i++)
    {
        projMat[i] = i % 5 ? 0.0f : 1.0f;
        viewMat[i] = i % 5 ? 0.0f : 1.0f;
    }
}

// -----------------------------------------------------------------------------------------------------------------
void Camera::resetView()
{
    if (up.z == 0.0f)
    {
        pos = Vec3(0.0f, 15.0f, 35.0f);
        forward = Vec3(0.0f, -0.3f, -1.0f);
        forward.normalize();
    }
    else
    {
        pos = Vec3(10.0f, 2.0f, 0.0f);
        forward = Vec3(-1.0f, 0.0f, 0.0f);
    }
    right = forward.cross(up);
}

// -----------------------------------------------------------------------------------------------------------------
void Camera::lookAt(const Vec3& _pos, const Vec3& at)
{
    pos = _pos;
    forward = at - pos;
    forward.normalize();
    up = Vec3(0.0f, 1.0f, 0.0f);
    right = forward.cross(up);
    right.normalize();
    up = right.cross(forward);
}

//------------------------------------------------------------------------------
void Camera::handleMouseView(int dx, int dy)
{
    forward.normalize();
    right = forward.cross(up);
    right.normalize();

    Quat qx(Pi * (-dx) * DEGS_PER_PIXEL / 180.0f, up);
    forward = qx.rotate(forward);
    Quat qy(Pi * (-dy) * DEGS_PER_PIXEL / 180.0f, right);
    forward = qy.rotate(forward);
}

//------------------------------------------------------------------------------
void Camera::handleMouseTranslate(int dx, int dy, float scale)
{
    pos -= right * scale * (float)dx;
    pos += up * scale * (float)dy;
}

//------------------------------------------------------------------------------
void Camera::handleMouseOrbit(int dx, int dy, const Vec3& center)
{
    Mat33 Q, R;
    Q.column0 = right;
    Q.column1 = forward;
    Q.column2 = up;

    up.normalize();
    right.normalize();

    Quat q = Quat(-dx * 0.01f, up);
    q = q * Quat(-dy * 0.01f, right);
    q.normalize();

    forward = q.rotate(forward);
    up = q.rotate(up);
    right = q.rotate(right);

    right.y = 0.0f; // force zero tilt
    right.normalize();
    up = right.cross(forward);
    up.normalize();
    forward = up.cross(right);

    R.column0 = right;
    R.column1 = forward;
    R.column2 = up;

    R = R * Q.getTranspose();

    pos = center + R * (pos - center);
}

//------------------------------------------------------------------------------
void Camera::handleKey(const bool keyDown[256])
{
    bool motion = false;
    if (keyDown['w'])
    {
        pos += forward * speed;
        motion = true;
    }
    if (keyDown['s'])
    {
        pos -= forward * speed;
        motion = true;
    }
    if (keyDown['a'])
    {
        pos -= right * speed;
        motion = true;
    }
    if (keyDown['d'])
    {
        pos += right * speed;
        motion = true;
    }
    if (keyDown['e'])
    {
        pos -= up * speed;
        motion = true;
    }
    if (keyDown['q'])
    {
        pos += up * speed;
        motion = true;
    }

    // if (motion)
    //	speed += 0.01f;
    // else
    //	speed = 0.1f;
}

//------------------------------------------------------------------------------
void Camera::handleWheel(int rotation)
{
    pos += (float)rotation * forward * speed;
}
