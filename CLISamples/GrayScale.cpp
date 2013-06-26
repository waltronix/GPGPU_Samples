// CLISamples.h

#include "Stdafx.h"

#include <cuda.h>
#include <cuda_runtime.h>

using namespace System;
using namespace Interfaces;

void grayScale(uchar4* pDataIn, uchar4* pDataOut, int width, int height, int blockDimX, int blockDimY);

namespace CLISamples 
{
    public ref class GrayScale : ImageFilter
    {
    protected:
        virtual array<Byte>^ PerformFilter(array<Byte>^ dataIn, int width, int height) override
        {
            array<Byte>^ dataOut = gcnew array<Byte>(4 * width * height);

            pin_ptr<Byte> pinnedDataIn = &dataIn[0];
            pin_ptr<Byte> pinnedDataIout = &dataOut[0];

            grayScale((uchar4*)pinnedDataIn, (uchar4*)pinnedDataIout, width, height,
                BlockDimX, BlockDimY);

            return dataOut;
        }
    };
}
