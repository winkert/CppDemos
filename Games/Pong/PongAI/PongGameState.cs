namespace TRW.Games.Pong.PongAI
{
    internal class PongGameState
    {
        internal int BallX { get; set; }
        internal int BallY { get; set; }
        internal int MyPaddleY { get; set; }
        internal bool BallHeadingToMe { get; set; }
        internal int BallSpeed { get; set; }
        /// <summary>
        /// Label
        /// </summary>
        internal bool MoveUp { get; set; }
    }
}
