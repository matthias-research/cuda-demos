#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glut.h>

// Forward declarations of CUDA kernels
extern "C" void launchParticleKernel(float4* pos, unsigned int width, unsigned int height, float time);
extern "C" void launchWaveKernel(uchar4* ptr, unsigned int width, unsigned int height, float time);
extern "C" void launchMandelbrotKernel(uchar4* ptr, unsigned int width, unsigned int height, float zoom, float centerX, float centerY);

// Window dimensions
const unsigned int WINDOW_WIDTH = 1024;
const unsigned int WINDOW_HEIGHT = 768;

// Current demo selection
enum DemoType { TEST, PARTICLES, WAVES, MANDELBROT };
DemoType currentDemo = TEST;

// OpenGL buffer object
GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource* cuda_pbo_resource;

// Animation
float animTime = 0.0f;
float animSpeed = 1.0f;
float zoom = 1.0f;
float centerX = -0.5f;
float centerY = 0.0f;

// UI state
bool showUI = true;
float fps = 0.0f;
int frameCount = 0;
float lastTime = 0.0f;

void initPixelBuffer() {
    // Create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * 4 * sizeof(GLubyte), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    // Register with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
    
    // Create texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void runCurrentDemo() {
    // Skip CUDA for test demo
    if (currentDemo == TEST) {
        return;
    }
    
    uchar4* d_out = 0;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, &num_bytes, cuda_pbo_resource);
    
    switch (currentDemo) {
        case PARTICLES:
            launchParticleKernel((float4*)d_out, WINDOW_WIDTH, WINDOW_HEIGHT, animTime);
            break;
        case WAVES:
            launchWaveKernel(d_out, WINDOW_WIDTH, WINDOW_HEIGHT, animTime);
            break;
        case MANDELBROT:
            launchMandelbrotKernel(d_out, WINDOW_WIDTH, WINDOW_HEIGHT, zoom, centerX, centerY);
            break;
        default:
            break;
    }
    
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display() {
    animTime += 0.01f * animSpeed;
    
    // Update FPS
    frameCount++;
    float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    if (currentTime - lastTime >= 1.0f) {
        fps = frameCount / (currentTime - lastTime);
        frameCount = 0;
        lastTime = currentTime;
    }
    
    // Clear to red for test demo, otherwise run CUDA
    if (currentDemo == TEST) {
        glClearColor(0.8f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    } else {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        runCurrentDemo();
        
        // Draw texture to screen
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    
    // ImGui rendering
    if (showUI) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGLUT_NewFrame();
        ImGui::NewFrame();
        
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(250, 400), ImGuiCond_FirstUseEver);
        ImGui::Begin("CUDA Demo Controls", &showUI, ImGuiWindowFlags_None);
        
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Separator();
        
        ImGui::Text("Demo Selection:");
        if (ImGui::Button("Test (Red Background)", ImVec2(200, 0))) {
            currentDemo = TEST;
        }
        if (ImGui::Button("Particle System", ImVec2(200, 0))) {
            currentDemo = PARTICLES;
        }
        if (ImGui::Button("Wave Simulation", ImVec2(200, 0))) {
            currentDemo = WAVES;
        }
        if (ImGui::Button("Mandelbrot Fractal", ImVec2(200, 0))) {
            currentDemo = MANDELBROT;
        }
        
        ImGui::Separator();
        ImGui::Text("Parameters:");
        ImGui::SliderFloat("Animation Speed", &animSpeed, 0.0f, 5.0f);
        
        if (currentDemo == MANDELBROT) {
            ImGui::SliderFloat("Zoom", &zoom, 0.1f, 1000.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Center X", &centerX, -2.0f, 2.0f);
            ImGui::SliderFloat("Center Y", &centerY, -2.0f, 2.0f);
            if (ImGui::Button("Reset View")) {
                zoom = 1.0f;
                centerX = -0.5f;
                centerY = 0.0f;
            }
        }
        
        ImGui::Separator();
        
        // GPU Info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        ImGui::Text("GPU: %s", prop.name);
        ImGui::Text("Compute: %d.%d", prop.major, prop.minor);
        
        ImGui::Separator();
        ImGui::Text("Press 'H' to hide/show UI");
        ImGui::Text("Press ESC to exit");
        
        ImGui::End();
        
        ImGui::Render();
        
        // Check if we have draw data
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (draw_data && draw_data->CmdListsCount > 0) {
            ImGui_ImplOpenGL3_RenderDrawData(draw_data);
        }
    }
    
    glutSwapBuffers();
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    // Check if ImGui context exists
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) return;
    }
    
    switch (key) {
        case '0':
            currentDemo = TEST;
            std::cout << "Demo: Test (Red Background)\n";
            break;
        case '1':
            currentDemo = PARTICLES;
            std::cout << "Demo: Particle System\n";
            break;
        case '2':
            currentDemo = WAVES;
            std::cout << "Demo: Wave Simulation\n";
            break;
        case '3':
            currentDemo = MANDELBROT;
            std::cout << "Demo: Mandelbrot Fractal\n";
            break;
        case 'h':
        case 'H':
            showUI = !showUI;
            std::cout << "UI " << (showUI ? "shown" : "hidden") << "\n";
            break;
        case '+':
        case '=':
            zoom *= 1.2f;
            break;
        case '-':
        case '_':
            zoom /= 1.2f;
            break;
        case 27: // ESC
            exit(0);
            break;
    }
}

void specialKeys(int key, int x, int y) {
    // Check if ImGui context exists
    if (ImGui::GetCurrentContext()) {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantCaptureKeyboard) return;
    }
    
    float moveSpeed = 0.1f / zoom;
    switch (key) {
        case GLUT_KEY_UP:
            centerY += moveSpeed;
            break;
        case GLUT_KEY_DOWN:
            centerY -= moveSpeed;
            break;
        case GLUT_KEY_LEFT:
            centerX -= moveSpeed;
            break;
        case GLUT_KEY_RIGHT:
            centerX += moveSpeed;
            break;
    }
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    ImGui_ImplGLUT_ReshapeFunc(w, h);
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
    std::cout << "CUDA + FreeGLUT Demos\n";
    std::cout << "=====================\n";
    std::cout << "Use the UI controls or keyboard shortcuts:\n";
    std::cout << "Press 0: Test (Red Background - no CUDA)\n";
    std::cout << "Press 1/2/3: CUDA demos\n";
    std::cout << "Press H: Hide/show UI\n";
    std::cout << "Press ESC: Exit\n\n";
    
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("CUDA Demos");
    
    // Initialize GLEW (must be after GLUT window creation)
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Initialize CUDA (cudaGLSetGLDevice is deprecated - CUDA auto-detects the device)
    
    // Initialize pixel buffer (before ImGui to ensure OpenGL context is ready)
    initPixelBuffer();
    
    // Setup ImGui (after OpenGL/GLEW initialization)
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // Disable imgui.ini
    ImGui::StyleColorsDark();
    
    // Initialize ImGui backends
    ImGui_ImplGLUT_Init();
    ImGui_ImplOpenGL3_Init("#version 130");
    
    // Setup callbacks (after ImGui is fully initialized)
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    glutMouseFunc(ImGui_ImplGLUT_MouseFunc);
    glutMotionFunc(ImGui_ImplGLUT_MotionFunc);
    glutPassiveMotionFunc(ImGui_ImplGLUT_MotionFunc);  // For hover effects
    
    std::cout << "Running Test demo (Red Background)...\n";
    
    // Main loop
    glutMainLoop();
    
    cleanup();
    return 0;
}

