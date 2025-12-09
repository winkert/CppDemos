namespace TRW.GameLibraries.GameCore
{
    public interface IPlayableGameObject
    {
        void KeyEvent(string keyPressed);
        Player Player { get; set; }
    }
}
