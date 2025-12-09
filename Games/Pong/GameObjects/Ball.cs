using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TRW.GameLibraries.GameCore;

namespace TRW.Games.Pong.GameObjects
{
    internal class Ball : PongGameObject
    {
        const double Acceleration = 0.01;

        internal Ball(double initialSpeed, double maxLeft, double maxTop)
            : base(maxLeft, maxTop, Statics.BallImage)
        {
            Speed = initialSpeed;
            // Math - Velocity of X and Y as random values between 0 and Speed result in random diagonal movement
            VelocityX = Statics.R.NextDouble() * Speed;
            VelocityY = Statics.R.NextDouble() * Speed;

        }

        public override string Name => "Ball";
        public override string Description => "Pong Ball";
        public override bool IsPlayable => false;

        internal double Speed { get; private set; }
        internal double VelocityX { get; private set; }
        internal double VelocityY { get; private set; }

        internal bool Bouncing { get; set; } = false;

        public override double Width => 30.0;
        public override double Height => 30.0;

        public override void GameTimerTick()
        {
            if (WpfImage != null)
            {
                WpfImage.Dispatcher.Invoke(new Action(() =>
                {
                    Left = System.Windows.Controls.Canvas.GetLeft(WpfImage);
                    Top = System.Windows.Controls.Canvas.GetTop(WpfImage);
                }));

                double currentLeft = Left;
                double currentTop = Top;

                double newLeft = currentLeft + VelocityX;
                double newTop = currentTop + VelocityY;

                if(newLeft >= LeftOuterBound || newLeft <= RightOuterBound)
                {
                    Contact(-1, 1);
                    newLeft = currentLeft + VelocityX;
                }
                if(newTop <= TopOuterBound || newTop >= BottomOuterBound)
                {
                    Contact(1, -1);
                    newTop = currentTop + VelocityY;
                }

                Left = newLeft;
                Top = newTop;

                WpfImage.Dispatcher.Invoke(new Action(() =>
                {
                    System.Windows.Controls.Canvas.SetLeft(WpfImage, Left);
                    System.Windows.Controls.Canvas.SetTop(WpfImage, Top);
                }));
            }
            Bouncing = false;
        }

        public void Contact(int xFlip, int yFlip)
        {
            if (Bouncing)
                return;

            Bouncing = true;
            // increase speed and adjust velocity accordingly
            Speed += Acceleration * Speed;
            double angle = Math.Atan2(VelocityY, VelocityX);
            VelocityX = Math.Cos(angle) * Speed * xFlip;
            VelocityY = Math.Sin(angle) * Speed * yFlip;
        }

    }
}
