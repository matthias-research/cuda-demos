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

//------------------------------------------------------------------------------
void Camera::setupMatrices(int width, int height)
{
    // Helper function to set perspective matrix
    auto setPerspective = [](float* mat, float fov, float aspect, float near, float far) {
        for (int i = 0; i < 16; i++) mat[i] = 0.0f;
        mat[0] = mat[5] = mat[10] = mat[15] = 1.0f;
        float f = 1.0f / tanf(fov * 0.5f);
        mat[0] = f / aspect;
        mat[5] = f;
        mat[10] = (far + near) / (near - far);
        mat[11] = -1.0f;
        mat[14] = (2.0f * far * near) / (near - far);
        mat[15] = 0.0f;
    };
    
    // Helper function to set lookAt matrix
    auto setLookAt = [](float* mat, float eyeX, float eyeY, float eyeZ,
                        float centerX, float centerY, float centerZ,
                        float upX, float upY, float upZ) {
        float fX = centerX - eyeX;
        float fY = centerY - eyeY;
        float fZ = centerZ - eyeZ;
        float len = sqrtf(fX*fX + fY*fY + fZ*fZ);
        fX /= len; fY /= len; fZ /= len;
        
        float sX = fY * upZ - fZ * upY;
        float sY = fZ * upX - fX * upZ;
        float sZ = fX * upY - fY * upX;
        len = sqrtf(sX*sX + sY*sY + sZ*sZ);
        sX /= len; sY /= len; sZ /= len;
        
        float uX = sY * fZ - sZ * fY;
        float uY = sZ * fX - sX * fZ;
        float uZ = (float)(sX * fY - sY * fX);
        
        mat[0] = sX;  mat[4] = sY;  mat[8] = sZ;   mat[12] = -(sX*eyeX + sY*eyeY + sZ*eyeZ);
        mat[1] = uX;  mat[5] = uY;  mat[9] = uZ;   mat[13] = -(uX*eyeX + uY*eyeY + uZ*eyeZ);
        mat[2] = -fX; mat[6] = -fY; mat[10] = -fZ; mat[14] = (fX*eyeX + fY*eyeY + fZ*eyeZ);
        mat[3] = 0;   mat[7] = 0;   mat[11] = 0;   mat[15] = 1;
    };
    
    // Compute view matrix
    Vec3 camTarget = pos + forward;
    setLookAt(viewMat,
             pos.x, pos.y, pos.z,
             camTarget.x, camTarget.y, camTarget.z,
             up.x, up.y, up.z);
    
    // Compute projection matrix
    setPerspective(projMat,
                  fov * 3.14159f / 180.0f,
                  (float)width / height,
                  nearClip, farClip);
}
