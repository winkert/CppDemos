namespace TRW.GameLibraries.GameCore
{
    public interface IGameObject
    {
        string Name { get; }
        string Description { get; }
        bool IsPlayable { get; }
        double X { get; }
        double Y { get; }
        double Width { get; }
        double Height { get; }
        void GameTimerTick();
        bool CollidesWith(IGameObject otherGameObject);
    }
}
