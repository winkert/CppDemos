using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using TRW.GameLibraries.GameCore;

namespace TRW.Games.Pong
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Tick = new GameTicker(100);
            uxMainMenu.MainWindow = this;
            MenuState = MenuStates.MainMenu;
            SetMenuVisible(true);
            SetMenuItems();
        }

        private GameTicker Tick { get; }
        private MenuStates MenuState { get; set; }
        private bool MenuVisible => MenuState is MenuStates.PauseMenu or MenuStates.SettingsMenu or MenuStates.MainMenu;

        #region Main Menu
        internal void CreateNewGame()
        {
            // reset board
            Tick.GameObjects.Clear();
            uxGameBoard.Children.Clear();

            // add ball to board
            GameObjects.Ball ball = new GameObjects.Ball(10, uxGameBoard.ActualWidth, uxGameBoard.ActualHeight);
            int objectIndx = uxGameBoard.Children.Add(ball.WpfImage);
            ball.ObjectId = objectIndx;
            Tick.GameObjects.Add(ball);
            Canvas.SetLeft(uxGameBoard.Children[objectIndx], uxGameBoard.ActualWidth / 2);
            Canvas.SetTop(uxGameBoard.Children[objectIndx], uxGameBoard.ActualHeight / 2);

            // add player 1 paddle to board
            GameObjects.Paddle paddle1 = new GameObjects.Paddle(8, uxGameBoard.ActualWidth, uxGameBoard.ActualHeight, ball) { Player = Player.PlayerOne };
            objectIndx = uxGameBoard.Children.Add(paddle1.WpfImage);
            paddle1.ObjectId = objectIndx;
            Tick.GameObjects.Add(paddle1);
            Canvas.SetLeft(paddle1.WpfImage, paddle1.Width);
            Canvas.SetTop(paddle1.WpfImage, (uxGameBoard.ActualHeight - paddle1.Height) / 2);

            // add player 2 paddle to board
            GameObjects.Paddle paddle2 = new GameObjects.Paddle(8, uxGameBoard.ActualWidth, uxGameBoard.ActualHeight, ball) { Player = Player.Computer };
            objectIndx = uxGameBoard.Children.Add(paddle2.WpfImage);
            paddle2.ObjectId = objectIndx;
            Tick.GameObjects.Add(paddle2);
            Canvas.SetLeft(paddle2.WpfImage, uxGameBoard.ActualWidth - paddle2.Width);
            Canvas.SetTop(paddle2.WpfImage, (uxGameBoard.ActualHeight - paddle1.Height) / 2);

            // clear menu and start playing
            Tick.GameStart();
            SetMenuVisible(false);
            SetMenuItems();


        }

        internal void ResumeGame()
        {
            SetMenuVisible(false);
            SetMenuItems();
            Tick.GameStart();
        }

        internal void PauseGame()
        {
            SetMenuVisible(true);
            SetMenuItems();
            Tick.GamePause();
        }

        internal void OpenSettings()
        {
            MenuState = MenuStates.SettingsMenu;
            SetMenuItems();
        }

        internal void ApplySettings()
        {
            if (Tick.GamePlaying)
            {
                MenuState = MenuStates.PauseMenu;
            }
            else
            {
                MenuState = MenuStates.MainMenu;
            }
            SetMenuItems();
        }

        internal void ExitGame()
        {
            Close();
        }

        private void SetMenuVisible(bool visible)
        {
            uxMainMenu.IsEnabled = visible;
            if (visible)
            {
                uxMainMenu.Visibility = Visibility.Visible;
            }
            else
            {
                uxMainMenu.Visibility = Visibility.Hidden;
                MenuState = MenuStates.Hidden;
            }

            Tick.GamePaused = visible;
        }

        private void SetMenuItems()
        {
            uxMainMenu.Buttons.Clear();
            switch (MenuState)
            {
                case MenuStates.MainMenu:
                    uxMainMenu.Buttons.Add(MainMenu.NewGameMenuItem);
                    uxMainMenu.Buttons.Add(MainMenu.SettingsMenuItem);
                    uxMainMenu.Buttons.Add(MainMenu.ExitGameMenuItem);
                    break;
                case MenuStates.SettingsMenu:
                    uxMainMenu.Buttons.Add(MainMenu.ApplySettingsMenuItem);
                    break;
                case MenuStates.PauseMenu:
                    uxMainMenu.Buttons.Add(MainMenu.ResumeGameMenuItem);
                    uxMainMenu.Buttons.Add(MainMenu.NewGameMenuItem);
                    uxMainMenu.Buttons.Add(MainMenu.SettingsMenuItem);
                    uxMainMenu.Buttons.Add(MainMenu.ExitGameMenuItem);
                    break;
            }


        }

        private void HandleMenuKey()
        {
            switch (MenuState)
            {
                case MenuStates.MainMenu:
                    break;
                case MenuStates.PauseMenu:
                    ResumeGame();
                    break;
                case MenuStates.SettingsMenu:
                case MenuStates.Hidden:
                    if (Tick.GamePlaying)
                    {
                        MenuState = MenuStates.PauseMenu;
                    }
                    else
                    {
                        MenuState = MenuStates.MainMenu;
                    }
                    PauseGame();
                    break;
            }
        }

        private void GamePlayKeyDown(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Escape:
                    HandleMenuKey();
                    break;
                case Key.W:
                    {
                        var paddle = Tick.GameObjects.OfType<GameObjects.Paddle>().FirstOrDefault(p => p.Player == Player.PlayerTwo);
                        if (paddle != null)
                        {
                            paddle.MovingUp = e.IsDown;
                        }
                    }
                    break;
                case Key.S:
                    {
                        var paddle = Tick.GameObjects.OfType<GameObjects.Paddle>().FirstOrDefault(p => p.Player == Player.PlayerTwo);
                        if (paddle != null)
                        {
                            paddle.MovingDown = e.IsDown;
                        }
                    }
                    break;
                case Key.Up:
                    {
                        var paddle = Tick.GameObjects.OfType<GameObjects.Paddle>().FirstOrDefault(p => p.Player == Player.PlayerOne);
                        if (paddle != null)
                        {
                            paddle.MovingUp = e.IsDown;
                        }
                    }
                    break;
                case Key.Down:
                    {
                        var paddle = Tick.GameObjects.OfType<GameObjects.Paddle>().FirstOrDefault(p => p.Player == Player.PlayerOne);
                        if (paddle != null)
                        {
                            paddle.MovingDown = e.IsDown;
                        }
                    }
                    break;
            }

            e.Handled = true;
        }

        #endregion

        private void uxGameBoard_KeyUp(object sender, KeyEventArgs e)
        {
            GamePlayKeyDown(sender, e);
        }

        private void uxMainMenu_KeyUp(object sender, KeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.Escape:
                    HandleMenuKey();
                    break;
            }

            e.Handled = true;
        }

        private void uxGameBoard_KeyDown(object sender, KeyEventArgs e)
        {
            GamePlayKeyDown(sender, e);
        }
    }

    internal enum MenuStates
    {
        Hidden,
        MainMenu,
        SettingsMenu,
        PauseMenu
    }
}
