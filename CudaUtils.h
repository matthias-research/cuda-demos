#pragma once

#include "Vec.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <vector>

#define cudaCheck(x)                                                                                                   \
    {                                                                                                                  \
        cudaAssert((x), __FILE__, __LINE__);                                                                           \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::string errorMsg =
            std::string("CUDA error: ") + cudaGetErrorString(code) + " at " + file + ":" + std::to_string(line);
        fprintf(stderr, "%s\n", errorMsg.c_str());
        throw std::runtime_error(errorMsg);
    }
}

#define THREADS_PER_BLOCK 256


template <typename T>
struct DeviceBuffer
{
    DeviceBuffer() : buffer(nullptr), size(0), capacity(0)
    {
    }

    void free()
    { // no destrcutor because cuda would call it in the kernel
        if (buffer)
            cudaCheck(cudaFree(buffer));
        capacity = 0;
        size = 0;
        buffer = nullptr;
    }

    void clear()
    {
        size = 0;
    }

    void resize(size_t newSize, bool preserveData = true)
    {
        if (newSize <= capacity)
        {
            size = newSize;
            return;
        }
    
        size_t newCapacity = size_t(newSize * 1.2f); 
        T* newBuffer;
    
        cudaCheck(cudaMalloc((void**)&newBuffer, newCapacity * sizeof(T)));
        
        if (buffer)
        {
            if (preserveData)
                cudaCheck(cudaMemcpy(newBuffer, buffer, size * sizeof(T), cudaMemcpyDeviceToDevice));
            cudaCheck(cudaFree(buffer));
        }
    
        buffer = newBuffer;
        size = newSize;
        capacity = newCapacity;
    }
    
    void setZero()
    {
        if (buffer)
            cudaCheck(cudaMemset(buffer, 0, size * sizeof(T)));
    }

    void set(const std::vector<T>& hostBuffer, int num = 0)
    {
        if (num == 0)
            num = (int)hostBuffer.size();
        if (num == 0)
            return;
        resize(num);
        cudaCheck(cudaMemcpy(buffer, hostBuffer.data(), num * sizeof(T), cudaMemcpyHostToDevice));
    }

    void set(const T* hostBuffer, size_t num)
    {
        if (num == 0)
        {
            size = 0;
            return;
        }
        resize(num);
        cudaCheck(cudaMemcpy(buffer, hostBuffer, num * sizeof(T), cudaMemcpyHostToDevice));
    }

    void get(std::vector<T>& hostBuffer, size_t num = 0)
    {
        if (num == 0)
            num = size;

        hostBuffer.resize(num);
        if (num == 0)
            return;
        cudaCheck(cudaMemcpy(hostBuffer.data(), buffer, num * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void setObject(int nr, const T& hostObject)
    {
        if (!buffer || nr >= size)
            return;
        cudaCheck(cudaMemcpy(buffer + nr, &hostObject, sizeof(T), cudaMemcpyHostToDevice));
    }

    void getDeviceObject(T& hostObject, int nr)
    {
        if (!buffer || nr >= size)
            return;

        cudaCheck(cudaMemcpy(&hostObject, buffer + nr, sizeof(T), cudaMemcpyDeviceToHost));
    }

    void append(const DeviceBuffer<T>& other)
    {
        if (other.size == 0)
            return;

        size_t oldSize = size;
        resize(size + other.size, true);

        cudaCheck(cudaMemcpy(buffer + oldSize, other.buffer, other.size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    size_t allocationSize() const
    {
        return capacity * sizeof(T);
    }

    void operator=(const DeviceBuffer& d)
    {
        if (!d.buffer)
            return;

        resize(d.size, false);
        cudaCheck(cudaMemcpy(buffer, d.buffer, d.size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    __host__ __device__ T& operator[](size_t i)
    {
        return buffer[i];
    }

    T* buffer;
    size_t size;
    size_t capacity;
};
