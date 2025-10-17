#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cmath>

// Forward declarations of CUDA kernels
extern "C" void launchParticleKernel(float4* pos, unsigned int width, unsigned int height, float time);
extern "C" void launchWaveKernel(uchar4* ptr, unsigned int width, unsigned int height, float time);
extern "C" void launchMandelbrotKernel(uchar4* ptr, unsigned int width, unsigned int height, float zoom, float centerX, float centerY);

// Window dimensions
const unsigned int WINDOW_WIDTH = 1024;
const unsigned int WINDOW_HEIGHT = 768;

// Current demo selection
enum DemoType { PARTICLES, WAVES, MANDELBROT };
DemoType currentDemo = PARTICLES;

// OpenGL buffer object
GLuint pbo = 0;
GLuint tex = 0;
struct cudaGraphicsResource* cuda_pbo_resource;

// Animation
float animTime = 0.0f;
float zoom = 1.0f;
float centerX = -0.5f;
float centerY = 0.0f;

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
    }
    
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void display() {
    animTime += 0.01f;
    
    runCurrentDemo();
    
    // Draw texture to screen
    glClear(GL_COLOR_BUFFER_BIT);
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
    
    glutSwapBuffers();
    glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
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

void cleanup() {
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &tex);
}

int main(int argc, char** argv) {
    std::cout << "CUDA + FreeGLUT Demos\n";
    std::cout << "=====================\n";
    std::cout << "Press 1: Particle System\n";
    std::cout << "Press 2: Wave Simulation\n";
    std::cout << "Press 3: Mandelbrot Fractal (use arrow keys to move, +/- to zoom)\n";
    std::cout << "Press ESC: Exit\n\n";
    
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("CUDA Demos");
    
    // Initialize CUDA
    cudaGLSetGLDevice(0);
    
    // Setup callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    
    // Initialize pixel buffer
    initPixelBuffer();
    
    std::cout << "Running Particle System demo...\n";
    
    // Main loop
    glutMainLoop();
    
    cleanup();
    return 0;
}

