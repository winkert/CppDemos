using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Media3D;
using TRW.GameLibraries.GameCore;

namespace TRW.Games.Pong.GameObjects
{
    internal abstract class PongGameObject: IVisibleGameObject
    {
        protected PongGameObject(double maxLeft, double maxTop, Bitmap objImage)
        {
            ObjectImage = objImage;

            /*
             *  0,0 ------------------>X,0
             * 
             *  0,Y ------------------>X,Y
             */

            LeftOuterBound = maxLeft - Width;
            TopOuterBound = 0;
            RightOuterBound = 0;
            BottomOuterBound = maxTop - Height;
        }

        System.Windows.Controls.Image? _wpfImage;
        public System.Windows.Controls.Image WpfImage
        {
            get
            {
                if (_wpfImage == null)
                {
                    _wpfImage = new System.Windows.Controls.Image() { Source = Statics.ToWpfImage(ObjectImage) };
                    _wpfImage.Width = Width;
                    _wpfImage.Height = Height;
                }
                return _wpfImage;
            }
        }

        public Bitmap ObjectImage { get; set; }
        public int ObjectId { get; set; }

        public abstract string Name { get; }
        public abstract string Description { get; }
        public abstract bool IsPlayable { get; }

        public abstract double Width { get; }
        public abstract double Height { get; }

        public int CollisionWidth => (int)(Width + Width / 10);
        public int CollisionHeight => (int)(Height + Height / 10);

        public double X => Left - Width / 2;
        public double Y => Top - Height / 2;

        internal protected double Left { get; set; }
        internal protected double Top { get; set; }

        internal protected double LeftOuterBound { get; private set; }
        internal protected double TopOuterBound { get; private set; }
        internal protected double RightOuterBound { get; private set; }
        internal protected double BottomOuterBound { get; private set; }

        public bool CollidesWith(IGameObject otherGameObject)
        {
            if (otherGameObject is IVisibleGameObject visibleGameObject)
            {
                return GameObject.IsColliding(this, visibleGameObject);
            }

            return false;
        }

        public abstract void GameTimerTick();
    }
}
