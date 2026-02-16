using System.Collections.Generic;
using System.Timers;

namespace TRW.GameLibraries.GameCore
{
    public class GameTicker
    {
        public Timer GameTimer { get; }
        public List<IGameObject> GameObjects { get; } = new List<IGameObject>();
        public bool GamePlaying { get; set; }
        public bool GamePaused { get; set; }
        public GameTicker(int elapsed)
        {
            GameTimer = new Timer(elapsed);
            GameTimer.Elapsed += GameTimer_Elapsed;
            GameTimer.Start();
        }

        public void GameStart()
        {
            GamePlaying = true;
            GamePaused = false;
            GameTimer.Start();
        }

        public void GamePause()
        {
            GamePaused = true;
            GamePlaying = false;
            GameTimer.Stop();
        }

        private void GameTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            if (GamePaused || !GamePlaying)
                return;

            GameTimer.Stop();

            // main game logic goes in here
            foreach (IGameObject gameObject in GameObjects)
            {
                gameObject.GameTimerTick();
            }

            GameTimer.Start();
        }
    }
}
