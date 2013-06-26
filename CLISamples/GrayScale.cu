#include "Stdafx.h"

// CUDA runtime
#include <cuda_runtime.h>

__global__ void grayScaleKernel(uchar4* pDataIn, uchar4* pDataOut, int width, int height)
{
    // get the position for the current thread
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    if(x < width && y < height)
    {
        // calculate the memory adress
        const unsigned int tid = y * width + x;

        // get the mean value for gray scaling
        unsigned char gv = (unsigned char)(1.0f / 3 * (pDataIn[tid].x + pDataIn[tid].y + pDataIn[tid].z));
        
        // write the value back to the global memory
        pDataOut[tid].x = gv;
        pDataOut[tid].y = gv;
        pDataOut[tid].z = gv;
        pDataOut[tid].w = 255;
    }
}

void grayScale(uchar4* pDataIn, uchar4* pDataOut, int width, int height, int blockDimX, int blockDimY)
{
    // allocate device memory
    unsigned int mem_size = sizeof(uchar4) * width * height;

    uchar4* pDevDataIn;
    uchar4* pDevDataOut;

    cudaError_t res;
    res = cudaMalloc((void **) &pDevDataIn, mem_size);
    res = cudaMalloc((void **) &pDevDataOut, mem_size);

    // copy results from host to device
    res = cudaMemcpy(pDevDataIn, pDataIn, mem_size, cudaMemcpyHostToDevice);

    // define partitioning
    dim3 threadsPerBlock(blockDimX, blockDimY);
    dim3 numBlocks(
        width / threadsPerBlock.x + 1, 
        height / threadsPerBlock.y + 1);

    // run the cuda kernel 
    grayScaleKernel<<<numBlocks, threadsPerBlock>>>(pDevDataIn, pDevDataOut, width, height);

    // copy results from device to host
    res = cudaMemcpy(pDataOut, pDevDataOut, mem_size, cudaMemcpyDeviceToHost);

    // cleanup memory
    res = cudaFree(pDevDataIn);
    res = cudaFree(pDevDataOut);
}
