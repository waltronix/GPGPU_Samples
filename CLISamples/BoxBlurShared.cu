#include <cuda_runtime.h>

const int CachDim = 64;

__global__ void boxBlurSharedKernel(uchar4* pDataIn, uchar4* pDataOut, 
    int width, int height, int borderSize)
{
    // get the position for the current thread
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    
    // calculate the memory adress
    const int tid = y * width + x;

    __shared__ uchar4 cache[CachDim][CachDim];

    uchar4 value = {127, 127, 127, 255};

    int meanLength = 2 * borderSize + 1;
    int denom = meanLength * meanLength;

    for(int cy = threadIdx.y; cy < blockDim.y + meanLength; cy += blockDim.y)
    for(int cx = threadIdx.x; cx < blockDim.x + meanLength; cx += blockDim.x)
    {
        int tmpX = (blockIdx.x * blockDim.x) + cx - borderSize;
        int tmpY = (blockIdx.y * blockDim.y) + cy - borderSize;

        if(tmpX >= 0 && tmpX < width
        && tmpY >= 0 && tmpY < height)
        {
            cache[cy][cx] = pDataIn[tmpY * width + tmpX];
        }
    }
    __syncthreads();

    if(x >= borderSize && y >= borderSize 
    && x + borderSize < width && y + borderSize < height)
    {
        int3 sum = {0, 0, 0};
    
        for(int cy = 0; cy < meanLength; cy++)
        for(int cx = 0; cx < meanLength; cx++)
        {
            uchar4 locValue = cache[threadIdx.y + cy][threadIdx.x + cx];

            sum.x += locValue.x;
            sum.y += locValue.y;
            sum.z += locValue.z;
        }

        value.x = (unsigned char)(sum.x / denom);
        value.y = (unsigned char)(sum.y / denom);
        value.z = (unsigned char)(sum.z / denom);
    }

    if(x < width && y < height)
    {
        // write the value back to the global memory
        pDataOut[tid] = value;
    }
}

void boxBlurShared(uchar4* pDataIn, uchar4* pDataOut, 
    int width, int height, int blurSize, int blockDimX, int blockDimY)
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
    dim3 numBlocks(width / threadsPerBlock.x + 1, height / threadsPerBlock.y + 1);

    // run the cuda kernel 
    boxBlurSharedKernel<<<numBlocks, threadsPerBlock>>>(pDevDataIn, pDevDataOut, 
        width, height, blurSize);

    // copy results from device to host
    res = cudaMemcpy(pDataOut, pDevDataOut, mem_size, cudaMemcpyDeviceToHost);

    // cleanup memory
    res = cudaFree(pDevDataIn);
    res = cudaFree(pDevDataOut);
}
