namespace TRW.GameLibraries.GameCore
{
    public static class GameObject
    {
        public static bool IsColliding(IVisibleGameObject obj1, IVisibleGameObject obj2)
        {
            return !(obj1.X + obj1.CollisionWidth < obj2.X ||
                     obj1.X > obj2.X + obj2.CollisionWidth ||
                     obj1.Y + obj1.CollisionHeight < obj2.Y ||
                     obj1.Y > obj2.Y + obj2.CollisionHeight);
        }
    }
}
