// CLISamples.h

#include "Stdafx.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace System;
using namespace Interfaces;

void boxBlurCPU(uchar4* pDataIn, uchar4* pDataOut, int width, int height, int borderSize)
{
    int denom = 2 * borderSize + 1;
    denom *= denom;
    
    for(int y = borderSize; y + borderSize < height; y++)
    for(int x = borderSize; x + borderSize < width; x++)
    {
        int3 sum = {0, 0, 0};
    
        for(int dy = -borderSize; dy <= borderSize; dy++)
        for(int dx = -borderSize; dx <= borderSize; dx++)
        {
            uchar4 locValue = pDataIn[(y + dy) * width + (x + dx)];
    
            sum.x += locValue.x;
            sum.y += locValue.y;
            sum.z += locValue.z;
        }
    
        uchar4 value = {127, 127, 127, 255};
        value.x = (unsigned char)(sum.x / denom);
        value.y = (unsigned char)(sum.y / denom);
        value.z = (unsigned char)(sum.z / denom);

        pDataOut[y * width + x] = value;
    }
}

namespace CLISamples 
{
    public ref class BoxBlurCPU : BoxBlurBase
    {
    protected:
        virtual array<Byte>^ PerformFilter(array<Byte>^ dataIn, int width, int height) override
        {
            array<Byte>^ dataOut = gcnew array<Byte>(4 * width * height);
            
            pin_ptr<Byte> pinnedDataIn = &dataIn[0];
            pin_ptr<Byte> pinnedDataOut = &dataOut[0];

            boxBlurCPU((uchar4*)pinnedDataIn, (uchar4*)pinnedDataOut, width, height, BlurSize);

            return dataOut;
        }
    };
}
