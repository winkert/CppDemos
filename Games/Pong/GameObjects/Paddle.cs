using System;
using TRW.GameLibraries.GameCore;

namespace TRW.Games.Pong.GameObjects
{
    internal class Paddle : PongGameObject, IPlayableGameObject
    {
        internal Paddle(double initialSpeed, double maxLeft, double maxTop, Ball ball)
            : base(maxLeft, maxTop, Statics.PaddleImage)
        {
            Speed = initialSpeed;
            VelocityY = Speed;

            _ball = ball;
        }

        internal readonly Ball _ball;

        public override string Name => "Paddle";

        public override string Description => "Pong Paddle";

        public override bool IsPlayable => true;
        public bool MovingUp { get; set; }
        public bool MovingDown { get; set; }

        internal double Speed { get; private set; }
        internal double VelocityX { get; private set; }
        internal double VelocityY { get; private set; }

        public Player Player { get; set; }

        public override double Width => 20.0;
        public override double Height => 100.0;

        public override void GameTimerTick()
        {
            if(CollidesWith(_ball))
            {
                _ball.Contact(-1, 1);
            }

            // if AI controlled, move paddle using neural network
            if(Player is Player.Computer)
            {
                double ballCenterY = _ball.Top + (_ball.Height / 2);
                double paddleCenterY = Top + (Height / 2);
                if(ballCenterY < paddleCenterY)
                {
                    MovePaddle(Speed * -1);
                }
                else if(ballCenterY > paddleCenterY)
                {
                    MovePaddle(Speed);
                }
            }
            else
            {
                if(MovingUp)
                {
                    MovePaddle(Speed * -1);
                }
                if(MovingDown)
                {
                    MovePaddle(Speed);
                }

            }
        }

        public void KeyEvent(string keyPressed)
        {
                switch(keyPressed)
                {
                    case "Up":
                        MovePaddle(Speed * -1);
                        break;
                    case "Down":
                        MovePaddle(Speed);
                        break;
                }
        }


        private void MovePaddle(double movement)
        {
            if (WpfImage != null)
            {
                
                WpfImage.Dispatcher.Invoke(new Action(() =>
                {
                    Left = System.Windows.Controls.Canvas.GetLeft(WpfImage);
                    Top = System.Windows.Controls.Canvas.GetTop(WpfImage);
                }));
                
                Top += movement;
                if (Top <= TopOuterBound)
                    Top = TopOuterBound;
                if (Top >= BottomOuterBound)
                    Top = BottomOuterBound;

                WpfImage.Dispatcher.Invoke(new Action(() =>
                {
                    System.Windows.Controls.Canvas.SetTop(WpfImage, Top);
                    System.Windows.Controls.Canvas.SetLeft(WpfImage, Left);
                }));
            }
        }
    }
}
