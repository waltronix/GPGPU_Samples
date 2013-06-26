#include <cuda_runtime.h>

texture<uchar4, 1, cudaReadModeElementType> texIn;

__global__ void boxBlurTexture1DKernel(uchar4* pData, 
    int width, int height, int borderSize)
{
    // get the position for the current thread
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    // calculate the memory adress
    const int tid = y * width + x;

    uchar4 value = {127, 127, 127, 255};

    int denom = 2 * borderSize + 1;
    denom *= denom;
    
    int3 sum = {0, 0, 0};
    
    for(int dy = -1 * borderSize; dy <= borderSize; dy++)
    for(int dx = -1 * borderSize; dx <= borderSize; dx++)
    {
        uchar4 locValue = tex1Dfetch(texIn, (y + dy) * width + (x + dx));

        sum.x += locValue.x;
        sum.y += locValue.y;
        sum.z += locValue.z;
    }

    value.x = (unsigned char)(sum.x / denom);
    value.y = (unsigned char)(sum.y / denom);
    value.z = (unsigned char)(sum.z / denom);
    
    if(x < width && y < height)
    {
        // write the value back to the global memory
        pData[tid] = value;
    }
}

void boxBlurTexture1D(uchar4* pDataIn, uchar4* pDataOut, int width, int height, int blurSize, int blockDimX, int blockDimY)
{
    // allocate device memory
    uchar4* pDevDataIn;
    uchar4* pDevDataOut;
    unsigned int mem_size = sizeof(uchar4) * width * height;

    cudaError_t res;
    res = cudaMalloc((void **) &pDevDataIn, mem_size);
    res = cudaMemcpy(pDevDataIn, pDataIn, mem_size, cudaMemcpyHostToDevice);
    res = cudaBindTexture(NULL, texIn, pDevDataIn, sizeof(uchar4) * width * height);

    res = cudaMalloc((void **) &pDevDataOut, mem_size);

    // define partitioning
    dim3 threadsPerBlock(blockDimX, blockDimY);
    dim3 numBlocks(width / threadsPerBlock.x + 1, height / threadsPerBlock.y + 1);

    // run the cuda kernel 
    boxBlurTexture1DKernel<<<numBlocks, threadsPerBlock>>>(
        pDevDataOut, width, height, blurSize);

    // copy results from device to host
    res = cudaMemcpy(pDataOut, pDevDataOut, mem_size, cudaMemcpyDeviceToHost);

    // cleanup memory
    res = cudaUnbindTexture(texIn);
    res = cudaFree(pDevDataIn);
    res = cudaFree(pDevDataOut);
}
