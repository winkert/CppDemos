using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;

namespace TRW.GameLibraries.GameCore
{
    public interface IVisibleGameObject : IGameObject
    {
        Bitmap ObjectImage { get; set; }
        int ObjectId { get; set; }
        int CollisionWidth { get; }
        int CollisionHeight { get; }
    }
}
