using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;
using System.Windows.Media.Imaging;
using System.Reflection;

namespace TRW.Games.Pong
{
    internal static class Statics
    {
        internal static Random R = new Random();
        internal static Bitmap BallImage = new Bitmap(GetResourceFromStream(Assembly.GetExecutingAssembly(), "TRW.Games.Pong.Images", "PongBall.gif"));
        internal static Bitmap PaddleImage = new Bitmap(GetResourceFromStream(Assembly.GetExecutingAssembly(), "TRW.Games.Pong.Images", "PongPaddle.gif"));

        internal static Stream GetResourceFromStream(Assembly assembly, string fullNamespace, string resourceFileName)
        {
            Stream? stream = assembly.GetManifestResourceStream(fullNamespace + "." + resourceFileName);
            return stream;
        }

        internal static BitmapImage ToWpfImage(Image img)
        {
            return ToWpfImage(img, img.Size.Width, img.Size.Height);
        }
        internal static BitmapImage ToWpfImage(Image img, int width, int height)
        {
            MemoryStream ms = new MemoryStream();  // no using here! BitmapImage will dispose the stream after loading
            img.Save(ms, System.Drawing.Imaging.ImageFormat.Gif);

            BitmapImage ix = new BitmapImage();
            ix.BeginInit();
            ix.CacheOption = BitmapCacheOption.OnLoad;
            ix.StreamSource = ms;
            ix.EndInit();
            return ix;
        }
    }
}
