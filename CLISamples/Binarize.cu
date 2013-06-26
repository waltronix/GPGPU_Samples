#include <cuda_runtime.h>

__global__ void binarizeKernel(uchar4* pData, unsigned char threshold)
{
    // get the position for the current thread
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    // calculate the memory adress
    const unsigned int tid = y * (gridDim.x * blockDim.x) + x;

    // get binarization result
    unsigned char value = pData[tid].x > threshold ? 255 : 0;
    
    // write the value back to the global memory
    pData[tid].x = value;
    pData[tid].y = value;
    pData[tid].z = value;
}

void binarize(uchar4* pDataIn, uchar4* pDataOut, int width, int height, unsigned char threshold)
{
    // allocate device memory
    uchar4* pDevData;
    unsigned int mem_size = sizeof(uchar4) * width * height;
    cudaMalloc((void **) &pDevData, mem_size);

    // copy results from host to device
    cudaMemcpy(pDevData, pDataIn, mem_size, cudaMemcpyHostToDevice);

    // define partitioning
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    // run the cuda kernel 
    binarizeKernel<<<numBlocks, threadsPerBlock>>>(pDevData, threshold);

    // copy results from device to host
    cudaMemcpy(pDataOut, pDevData, mem_size, cudaMemcpyDeviceToHost);

    // cleanup memory
    cudaFree(pDevData);
}
