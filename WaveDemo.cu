#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

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

