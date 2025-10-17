#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Particle System Demo
__global__ void particleKernel(float4* pos, unsigned int width, unsigned int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x >= width || y >= height) return;
    
    float fx = x / (float)width;
    float fy = y / (float)height;
    
    // Create multiple particle streams
    float r = 0.0f, g = 0.0f, b = 0.0f;
    
    for (int i = 0; i < 5; i++) {
        float angle = time * 0.3f + i * 2.0f;
        float px = 0.5f + 0.3f * cosf(angle);
        float py = 0.5f + 0.3f * sinf(angle);
        
        float dx = fx - px;
        float dy = fy - py;
        float dist = sqrtf(dx * dx + dy * dy);
        
        float intensity = expf(-dist * 20.0f);
        r += intensity * (0.5f + 0.5f * sinf(time + i));
        g += intensity * (0.5f + 0.5f * cosf(time * 0.7f + i));
        b += intensity * (0.5f + 0.5f * sinf(time * 1.3f + i));
    }
    
    pos[idx] = make_float4(
        fminf(r * 255.0f, 255.0f),
        fminf(g * 255.0f, 255.0f),
        fminf(b * 255.0f, 255.0f),
        255.0f
    );
}

extern "C" void launchParticleKernel(float4* pos, unsigned int width, unsigned int height, float time) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    particleKernel<<<grid, block>>>(pos, width, height, time);
    cudaDeviceSynchronize();
}

// Wave Simulation Demo
__global__ void waveKernel(uchar4* ptr, unsigned int width, unsigned int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x >= width || y >= height) return;
    
    float fx = (x - width / 2.0f) / (width / 2.0f);
    float fy = (y - height / 2.0f) / (height / 2.0f);
    
    float dist = sqrtf(fx * fx + fy * fy);
    float wave = sinf(dist * 10.0f - time * 3.0f) * 0.5f + 0.5f;
    float wave2 = sinf(dist * 15.0f + time * 2.0f) * 0.5f + 0.5f;
    
    float value = (wave + wave2) * 0.5f;
    
    unsigned char r = (unsigned char)(value * 255.0f);
    unsigned char g = (unsigned char)((1.0f - value) * wave * 255.0f);
    unsigned char b = (unsigned char)((1.0f - value) * wave2 * 255.0f);
    
    ptr[idx] = make_uchar4(r, g, b, 255);
}

extern "C" void launchWaveKernel(uchar4* ptr, unsigned int width, unsigned int height, float time) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    waveKernel<<<grid, block>>>(ptr, width, height, time);
    cudaDeviceSynchronize();
}

// Mandelbrot Fractal Demo
__global__ void mandelbrotKernel(uchar4* ptr, unsigned int width, unsigned int height, float zoom, float centerX, float centerY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x >= width || y >= height) return;
    
    float jx = (x - width / 2.0f) / (width / 4.0f) / zoom + centerX;
    float jy = (y - height / 2.0f) / (height / 4.0f) / zoom + centerY;
    
    float cx = jx;
    float cy = jy;
    
    int iter = 0;
    int maxIter = 256;
    
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

