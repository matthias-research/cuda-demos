#pragma once

#include "Vec.h"


// Camera
// by Matthias Müller

// ---------------------------------------------------------------------------------------
class Camera
{
public:
    void init();
    void resetView();
    void lookAt(const Vec3& pos, const Vec3& at);
    void setUpY()
    {
        up = Vec3(0.0f, 1.0f, 0.0f);
        resetView();
    }
    void setUpZ()
    {
        up = Vec3(0.0f, 0.0f, 1.0f);
        resetView();
    }
    void handleMouseView(int dx, int dy);
    void handleMouseOrbit(int dx, int dy, const Vec3& center);
    void handleMouseTranslate(int dx, int dy, float scale = 0.01f);
    void handleKey(const bool keyDown[256]);
    void handleWheel(int rotation);
    void setupMatrices(int width, int height);

    Vec3 pos;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float speed;
    float fov;

    float projMat[16];
    float viewMat[16];
};
