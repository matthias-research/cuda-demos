#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Particle System Demo
__global__ void particleKernel(uchar4* ptr, unsigned int width, unsigned int height, float time) {
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
    
    ptr[idx] = make_uchar4(
        (unsigned char)(fminf(r * 255.0f, 255.0f)),
        (unsigned char)(fminf(g * 255.0f, 255.0f)),
        (unsigned char)(fminf(b * 255.0f, 255.0f)),
        255
    );
}

extern "C" void launchParticleKernel(uchar4* ptr, unsigned int width, unsigned int height, float time) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    particleKernel<<<grid, block>>>(ptr, width, height, time);
    cudaDeviceSynchronize();
}

