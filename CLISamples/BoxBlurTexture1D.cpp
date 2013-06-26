// CLISamples.h

#include "Stdafx.h"

using namespace System;
using namespace Interfaces;

void boxBlurTexture1D(uchar4* pDataIn, uchar4* pDataOut, int width, int height, int blurSize, int blockDimX, int blockDimY);

namespace CLISamples 
{
    public ref class BoxBlurTexture1D : BoxBlurBase
    {
    protected:
        virtual array<Byte>^ PerformFilter(array<Byte>^ dataIn, int width, int height) override
        {
            array<Byte>^ dataOut = gcnew array<Byte>(4 * width * height);
            
            pin_ptr<Byte> pinnedDataIn = &dataIn[0];
            pin_ptr<Byte> pinnedDataIout = &dataOut[0];

            boxBlurTexture1D((uchar4*)pinnedDataIn, (uchar4*)pinnedDataIout, width, height, 
                BlurSize, BlockDimX, BlockDimY);

            return dataOut;
        }
    };
}
