using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TRW.Games.Pong.UI
{
    internal enum ButtonState
    {
        Normal,
        Hover,
        Pressed
    }

    internal enum MenuStates
    {
        Hidden,
        MainMenu,
        SettingsMenu,
        PauseMenu
    }

    internal class Button
    {
        internal Button(float x, float y, float width, float height, string text, Action onClick)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
            Text = text;
            OnClick = onClick;
            State = ButtonState.Normal;
        }
        internal float X, Y, Width, Height;
        internal string Text;
        internal ButtonState State;
        internal Action OnClick;

        internal float R { get
            {
                var (r, _, _) = GetColor();
                return r;
            }
        }
        internal float G
        {
            get
            {
                var (_, g, _) = GetColor();
                return g;
            }
        }
        internal float B
        {
            get
            {
                var (_, _, b) = GetColor();
                return b;
            }
        }

        internal (float R, float G, float B) GetColor()
        {
            return State switch
            {
                ButtonState.Normal => (0.25f, 0.25f, 0.25f),
                ButtonState.Hover => (1f, 1f, 1f),
                ButtonState.Pressed => (0.59f, 0.59f, 0.59f),
                _ => (0.8f, 0.8f, 0.8f)
            };
        }

        internal bool Contains(float mx, float my)
        {
            return mx >= X && mx <= X + Width && my >= Y && my <= Y + Height;
        }
    }

    internal class Screen
    {
        internal MenuStates ScreenState;
        internal List<Button> Buttons = new();
    }
}
