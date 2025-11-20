#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <memory>
#include <vector>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glut.h>

#include "Demo.h"
#include "Camera.h"
#include "BoxesDemo.h"
#include "MandelbrotDemo.h"
#include "BallsDemo.h"

// Window dimensions (can change on resize)
unsigned int windowWidth = 1024;
unsigned int windowHeight = 768;

// Demo management
std::vector<std::unique_ptr<Demo>> demos;
int currentDemoIndex = 0;

// Camera for 3D demos
Camera camera;
bool keyDown[256] = {false};
int lastMouseX = 0;
int lastMouseY = 0;
bool isMouseDragging = false;
bool isRightMouseDragging = false;
bool isMiddleMouseDragging = false;
Vec3 orbitCenter(0.0f, 0.0f, 0.0f);  // Center point for camera orbit

// OpenGL buffer object
GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource* cuda_pbo_resource;

// UI state
bool showUI = true;
float fps = 0.0f;
int frameCount = 0;
float lastTime = 0.0f;

// GPU info (cached at startup)
std::string gpuName;

void initPixelBuffer() {
    if (pbo != 0) {
        // Clean up old buffer if it exists
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    
    // Create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, windowWidth * windowHeight * 4 * sizeof(GLubyte), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    // Register with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    
    // Create texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void display() {
    // Calculate delta time
    static float lastFrameTime = 0.0f;
    float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    float deltaTime = currentTime - lastFrameTime;
    deltaTime = 1.0f / 30.0f; // fixed frame rate
    lastFrameTime = currentTime;
    
    // Update FPS
    frameCount++;
    if (currentTime - lastTime >= 1.0f) {
        fps = frameCount / (currentTime - lastTime);
        frameCount = 0;
        lastTime = currentTime;
    }
    
    // Update current demo
    demos[currentDemoIndex]->update(deltaTime);
    
    // Render CUDA demo
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Get CUDA device pointer
    uchar4* d_out = 0;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, &num_bytes, cuda_pbo_resource);
    
    // Render current demo
    demos[currentDemoIndex]->render(d_out, windowWidth, windowHeight);
    
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
    
    // Draw texture to screen
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    // ImGui rendering
    if (showUI) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGLUT_NewFrame();
        ImGui::NewFrame();
        
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(320, 750), ImGuiCond_FirstUseEver);
        ImGui::Begin("CUDA Demo Controls", &showUI, ImGuiWindowFlags_None);
        
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("GPU: %s", gpuName.c_str());
        ImGui::Separator();
        
        ImGui::Text("Demo Selection:");
        for (int i = 0; i < demos.size(); i++) {
            if (ImGui::Button(demos[i]->getName(), ImVec2(200, 0))) {
                currentDemoIndex = i;
                std::cout << "Switched to: " << demos[i]->getName() << "\n";
            }
        }
        
        ImGui::Separator();
        ImGui::Text("Controls:");
        
        // Render current demo's UI
        demos[currentDemoIndex]->renderUI();
        
        ImGui::End();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
    
    glutSwapBuffers();
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    // Let ImGui handle input first
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) return;
    }
    
    switch (key) {
        case '1':
            if (currentDemoIndex != 0) {
                currentDemoIndex = 0;
                std::cout << "Switched to: " << demos[0]->getName() << "\n";
            }
            break;
        case '2':
            if (currentDemoIndex != 1) {
                currentDemoIndex = 1;
                std::cout << "Switched to: " << demos[1]->getName() << "\n";
            }
            break;
        case '3':
            if (currentDemoIndex != 2) {
                currentDemoIndex = 2;
                std::cout << "Switched to: " << demos[2]->getName() << "\n";
            }
            break;
        case 'h':
        case 'H':
            showUI = !showUI;
            std::cout << "UI " << (showUI ? "shown" : "hidden") << "\n";
            break;
        case 27: // ESC
            exit(0);
            break;
        default:
            // Track keys for camera (3D demos)
            if (key < 256) {
                keyDown[key] = true;
                // Handle camera movement immediately for 3D demos
                if (demos[currentDemoIndex]->is3D()) {
                    camera.handleKey(keyDown);
                }
            }
            // Pass to current demo
            demos[currentDemoIndex]->onKeyPress(key);
            break;
    }
}

void keyboardUp(unsigned char key, int x, int y) {
    if (key < 256) {
        keyDown[key] = false;
        // Update camera state immediately for 3D demos
        if (demos[currentDemoIndex]->is3D()) {
            camera.handleKey(keyDown);
        }
    }
}

void specialKeys(int key, int x, int y) {
    // Let ImGui handle input first
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) return;
    }
    
    // Pass to current demo (arrow keys, etc.)
    demos[currentDemoIndex]->onSpecialKey(key);
}

// Helper function to compute ray from screen coordinates
void getMouseRay(int x, int y, Vec3& orig, Vec3& dir) {
    GLint viewport[4];
    GLdouble modelMatrix[16];
    GLdouble projMatrix[16];
    
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
    glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
    
    // Flip Y coordinate
    int flippedY = viewport[3] - y - 1;
    
    GLdouble ox, oy, oz;
    GLdouble dx, dy, dz;
    
    // Unproject near plane
    gluUnProject((GLdouble)x, (GLdouble)flippedY, 0.0, modelMatrix, projMatrix, viewport, &ox, &oy, &oz);
    // Unproject far plane
    gluUnProject((GLdouble)x, (GLdouble)flippedY, 1.0, modelMatrix, projMatrix, viewport, &dx, &dy, &dz);
    
    orig = Vec3((float)ox, (float)oy, (float)oz);
    Vec3 farPoint((float)dx, (float)dy, (float)dz);
    dir = farPoint - orig;
    dir.normalize();
}

void mouse(int button, int state, int x, int y) {
    // Let ImGui handle input first
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGLUT_MouseFunc(button, state, x, y);
        if (io.WantCaptureMouse) return;
    }
    
    // Handle camera input for 3D demos
    if (demos[currentDemoIndex]->is3D()) {
        if (button == GLUT_LEFT_BUTTON) {
            if (state == GLUT_DOWN) {
                // Raycast to find orbit center
                Vec3 rayOrig, rayDir;
                getMouseRay(x, y, rayOrig, rayDir);
                
                float t;
                if (demos[currentDemoIndex]->raycast(rayOrig, rayDir, t)) {
                    // Hit! Use the hit point as orbit center
                    orbitCenter = rayOrig + rayDir * t;
                } else {
                    // No hit, use origin as default
                    orbitCenter = Vec3(0.0f, 0.0f, 0.0f);
                }
                
                isMouseDragging = true;
                lastMouseX = x;
                lastMouseY = y;
            } else {
                isMouseDragging = false;
            }
        }
        else if (button == GLUT_MIDDLE_BUTTON) {
            if (state == GLUT_DOWN) {
                isMiddleMouseDragging = true;
                lastMouseX = x;
                lastMouseY = y;
            } else {
                isMiddleMouseDragging = false;
            }
        }
        else if (button == GLUT_RIGHT_BUTTON) {
            if (state == GLUT_DOWN) {
                isRightMouseDragging = true;
                lastMouseX = x;
                lastMouseY = y;
            } else {
                isRightMouseDragging = false;
            }
        }
    }
    
    // Pass to current demo
    demos[currentDemoIndex]->onMouseClick(button, state, x, y);
}

void motion(int x, int y) {
    // Let ImGui handle input first
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        ImGui_ImplGLUT_MotionFunc(x, y);
        if (io.WantCaptureMouse) return;
    }
    
    // Handle camera input for 3D demos
    if (demos[currentDemoIndex]->is3D()) {
        int dx = x - lastMouseX;
        int dy = y - lastMouseY;
        
        if (isMouseDragging) {
            // Left button = orbit around raycast hit point (or origin)
            camera.handleMouseOrbit(dx, dy, orbitCenter);
        }
        else if (isMiddleMouseDragging) {
            // Middle button = translate/pan (scale adapts to distance from origin)
            float scale = camera.pos.magnitude() * 0.001f;
            camera.handleMouseTranslate(dx, dy, scale);
        }
        else if (isRightMouseDragging) {
            // Right button = first-person view rotation
            camera.handleMouseView(dx, dy);
        }
        
        lastMouseX = x;
        lastMouseY = y;
    }
    
    // Pass to current demo
    demos[currentDemoIndex]->onMouseDrag(x, y);
}

void mouseWheel(int wheel, int direction, int x, int y) {
    // Let ImGui handle input first
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureMouse) return;
    }
    
    // Handle camera input for 3D demos
    if (demos[currentDemoIndex]->is3D()) {
        camera.handleWheel(direction);
    }
    
    // Pass to current demo
    demos[currentDemoIndex]->onMouseWheel(wheel, direction, x, y);
}

void reshape(int w, int h) {
    if (h == 0) h = 1; // Prevent divide by zero
    
    windowWidth = w;
    windowHeight = h;
    
    glViewport(0, 0, w, h);
    ImGui_ImplGLUT_ReshapeFunc(w, h);
    
    // Recreate pixel buffer with new size
    initPixelBuffer();
}

void cleanup() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGLUT_Shutdown();
    ImGui::DestroyContext();
    
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
}

int main(int argc, char** argv) {
    std::cout << "CUDA + OpenGL Demos\n";
    std::cout << "===================\n";
    std::cout << "Controls:\n";
    std::cout << "  1: 3D Boxes (OpenGL)\n";
    std::cout << "  2: Mandelbrot Fractal (CUDA)\n";
    std::cout << "  3: 3D Bouncing Balls (OpenGL)\n";
    std::cout << "  H: Hide/show UI\n";
    std::cout << "  ESC: Exit\n\n";
    std::cout << "3D Camera (Boxes and Balls demos):\n";
    std::cout << "  WASD: Move, Q/E: Up/Down\n";
    std::cout << "  Left Mouse: Orbit around point\n";
    std::cout << "  Middle Mouse: Pan/Translate\n";
    std::cout << "  Right Mouse: Rotate view\n";
    std::cout << "  Mouse wheel: Zoom\n\n";
    std::cout << "Mandelbrot controls:\n";
    std::cout << "  Mouse wheel: Zoom\n";
    std::cout << "  Click and drag: Pan\n";
    std::cout << "  Arrow keys: Pan\n\n";
    
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("CUDA Demos");
    
    // Initialize GLEW (must be after GLUT window creation)
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Get GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    gpuName = prop.name;
    std::cout << "Using GPU: " << gpuName << "\n\n";
    
    // Initialize pixel buffer
    initPixelBuffer();
    
    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    
    ImGui_ImplGLUT_Init();
    ImGui_ImplOpenGL3_Init("#version 130");
    
    // Custom dark style with orange accents and transparency
    ImGuiStyle& style = ImGui::GetStyle();
    style.Alpha = 1.0f;
    style.WindowRounding = 4.0f;
    style.FrameRounding = 2.0f;
    style.GrabRounding = 2.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                   = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.08f, 0.08f, 0.08f, 0.90f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.08f, 0.08f, 0.08f, 0.94f);
    colors[ImGuiCol_Border]                 = ImVec4(0.30f, 0.30f, 0.30f, 0.50f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.15f, 0.15f, 0.15f, 0.90f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.90f, 0.50f, 0.20f, 0.40f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.90f, 0.50f, 0.20f, 0.67f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.90f, 0.50f, 0.20f, 0.80f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.90f, 0.50f, 0.20f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.90f, 0.50f, 0.20f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.95f, 0.60f, 0.30f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.90f, 0.50f, 0.20f, 0.60f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(1.00f, 0.60f, 0.30f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.90f, 0.50f, 0.20f, 0.55f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.95f, 0.55f, 0.25f, 0.80f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_Separator]              = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.90f, 0.50f, 0.20f, 0.78f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.90f, 0.50f, 0.20f, 0.25f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.95f, 0.55f, 0.25f, 0.67f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.95f, 0.60f, 0.30f, 0.95f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.90f, 0.50f, 0.20f, 0.86f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.95f, 0.55f, 0.25f, 0.80f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.07f, 0.10f, 0.15f, 0.97f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.14f, 0.26f, 0.42f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.90f, 0.50f, 0.20f, 0.35f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(0.95f, 0.55f, 0.25f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
    
    // Initialize camera
    camera.init();
    camera.resetView();
    camera.speed = 1.0f;
    
    // Create demos
    demos.push_back(std::make_unique<BoxesDemo>());
    demos.push_back(std::make_unique<MandelbrotDemo>());
    demos.push_back(std::make_unique<BallsDemo>());
    
    // Set camera for 3D demos
    for (auto& demo : demos) {
        if (demo->is3D()) {
            demo->setCamera(&camera);
        }
    }
    
    std::cout << "Starting with: " << demos[0]->getName() << "\n\n";
    
    // Setup callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutKeyboardUpFunc(keyboardUp);
    glutSpecialFunc(specialKeys);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(ImGui_ImplGLUT_MotionFunc);
    glutMouseWheelFunc(mouseWheel);
    
    // Main loop
    glutMainLoop();
    
    cleanup();
    return 0;
}
