using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;
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

        internal static class MainMenu
        {
            internal const string NewGameMenuItem = "New Game";
            internal const string SettingsMenuItem = "Settings";
            internal const string ExitMenuItem = "Exit";
            internal const string EnableTrainingMode = "Enable Training Mode";
            internal const string UseTrainedAIOpponent = "Use Trained AI Opponent";
            internal const string ApplySettingsMenuItem = "Apply Settings";

        }
    }
}
