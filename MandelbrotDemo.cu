#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Mandelbrot Fractal Demo
__global__ void mandelbrotKernel(uchar4* ptr, unsigned int width, unsigned int height, float zoom, float centerX, float centerY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x >= width || y >= height) return;
    
    // Map pixel coordinates to complex plane with correct aspect ratio
    // Normalize both axes by height/2 to ensure square pixels
    float scale = 2.0f / zoom;
    float jx = ((x - width / 2.0f) / (height / 2.0f)) * scale + centerX;
    float jy = ((y - height / 2.0f) / (height / 2.0f)) * scale + centerY;
    
    float cx = jx;
    float cy = jy;
    
    int iter = 0;
    // Dynamic iteration count based on zoom level
    // More zoom = more detail = more iterations needed
    int maxIter = 256 + (int)(log2f(fmaxf(zoom, 1.0f)) * 32.0f);
    
    while (cx * cx + cy * cy < 4.0f && iter < maxIter) {
        float xtemp = cx * cx - cy * cy + jx;
        cy = 2.0f * cx * cy + jy;
        cx = xtemp;
        iter++;
    }
    
    unsigned char r, g, b;
    if (iter == maxIter) {
        r = g = b = 0;
    } else {
        float t = (float)iter / maxIter;
        r = (unsigned char)(9.0f * (1.0f - t) * t * t * t * 255.0f);
        g = (unsigned char)(15.0f * (1.0f - t) * (1.0f - t) * t * t * 255.0f);
        b = (unsigned char)(8.5f * (1.0f - t) * (1.0f - t) * (1.0f - t) * t * 255.0f);
    }
    ptr[idx] = make_uchar4(r, g, b, 255);
}

extern "C" void launchMandelbrotKernel(uchar4* ptr, unsigned int width, unsigned int height, float zoom, float centerX, float centerY) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    mandelbrotKernel<<<grid, block>>>(ptr, width, height, zoom, centerX, centerY);
    cudaDeviceSynchronize();
}

