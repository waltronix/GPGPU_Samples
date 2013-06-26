namespace Interfaces
{
    using System;
    using System.Diagnostics;
    using System.Drawing;
    using System.Drawing.Imaging;
    using System.Runtime.InteropServices;

    public abstract class ImageFilter
    {
        public int BlockDimX = 32;
        public int BlockDimY = 32;

        public Bitmap Run(Bitmap bmpIn)
        {
            BitmapData bmpData = bmpIn.LockBits(new Rectangle(0, 0, bmpIn.Width, bmpIn.Height),
               ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            int length = bmpData.Stride * bmpData.Height;

            byte[] dataIn = new byte[length];
            Marshal.Copy(bmpData.Scan0, dataIn, 0, length);
            bmpIn.UnlockBits(bmpData);

            byte[] dataOut = null;

            Stopwatch watch = Stopwatch.StartNew();

            dataOut = PerformFilter(dataIn, bmpIn.Width, bmpIn.Height);

            watch.Stop();
            Console.WriteLine(this.GetType().Name + ".Run(...) took " + watch.ElapsedMilliseconds + " ms");

            Bitmap bmpOut = new Bitmap(bmpIn.Width, bmpIn.Height, PixelFormat.Format32bppArgb);
            bmpData = bmpOut.LockBits(new Rectangle(0, 0, bmpOut.Width, bmpOut.Height),
               ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            
            Marshal.Copy(dataOut, 0, bmpData.Scan0, length);
            bmpOut.UnlockBits(bmpData);

            return bmpOut;
        }

        protected abstract byte[] PerformFilter(byte[] dataIn, int width, int height);
    }
}
