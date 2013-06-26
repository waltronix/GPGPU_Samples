namespace CmdLineRunner
{
    using System;
    using System.Diagnostics;
    using System.Drawing;
    using System.IO;
    using CLISamples;
    using Interfaces;

    class Program
    {
        static void Main(string[] args)
        {
            string filePath = "TestImages/Märzenbecher.png";
            string pathOut = Path.GetFileNameWithoutExtension(filePath);

            Bitmap bmpIn = new Bitmap(filePath);
            Bitmap bmpOut = null;

            ImageFilter grayScale = new GrayScale();
            bmpOut = grayScale.Run(bmpIn);
            bmpOut = grayScale.Run(bmpIn);
            bmpOut.Save(pathOut + ".GrayScale.bmp");

            ImageFilter blurCPU = new BoxBlurCPU();
            bmpOut = blurCPU.Run(bmpIn);
            bmpOut.Save(pathOut + ".BoxBlurCPU.bmp");

            ImageFilter blurGPU = new BoxBlur();
            bmpOut = blurGPU.Run(bmpIn);
            bmpOut.Save(pathOut + ".BoxBlur.bmp");

            ImageFilter blurTexture = new BoxBlurTexture1D();
            bmpOut = blurTexture.Run(bmpIn);
            bmpOut.Save(pathOut + ".BoxBlurTexture1D.bmp");

            ImageFilter blurShared = new BoxBlurShared();
            bmpOut = blurShared.Run(bmpIn);
            bmpOut.Save(pathOut + ".BoxBlurShared.bmp");

            ImageFilter blurSplit = new BoxBlurSplit();
            bmpOut = blurSplit.Run(bmpIn);
            bmpOut.Save(pathOut + ".BoxBlurSplit.bmp");

            if (Debugger.IsAttached)
            {
                Console.ReadLine();
            }
        }
    }
}
