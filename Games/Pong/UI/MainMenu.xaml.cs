using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace TRW.Games.Pong
{
    /// <summary>
    /// Interaction logic for MainMenu.xaml
    /// </summary>
    public partial class MainMenu : UserControl
    {
        public const string NewGameMenuItem = "New Game";
        public const string ResumeGameMenuItem = "Resume Game";
        public const string SettingsMenuItem = "Settings";

        public const string ApplySettingsMenuItem = "Apply Settings";
        public const string ExitGameMenuItem = "Exit";


        public MainMenu()
        {
            InitializeComponent();
            uxGridSource.ItemsSource = Buttons;
        }

        public ObservableCollection<string> Buttons { get; } = new ObservableCollection<string>();
        public MainWindow? MainWindow { get; internal set; }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Button? button = sender as Button;

            if (button == null || MainWindow == null)
                return;

            switch (button.Content)
            {
                case NewGameMenuItem:
                    MainWindow.CreateNewGame();
                    break;
                case ResumeGameMenuItem:
                    MainWindow.ResumeGame();
                    break;
                case SettingsMenuItem:
                    MainWindow.OpenSettings();
                    break;
                case ApplySettingsMenuItem:
                    MainWindow.ApplySettings();
                    break;
                case ExitGameMenuItem:
                    MainWindow.ExitGame();
                    break;
                default:
                    return;
            }
        }
    }

}
