using Silk.NET.Input;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using TRW.GameLibraries.GameCore;
using TRW.Games.Pong.UI;

namespace TRW.Games.Pong
{
    public static class Program
    {
        const int Width = 800;
        const int Height = 600;

        static Screen currentScreen;
        static Screen mainMenuScreen;
        static Screen settingsScreen;
        static GL GlContext;
        static IWindow PongWindow;
        static IMouse Mouse;
        static IKeyboard Keyboard;

        internal static uint UIShaderProgram;
        internal static uint Vbo;
        internal static uint Ebo;
        internal static uint QuadVao;
        internal static uint UScreenSizeLoc;
        internal static uint UPositionLoc;
        internal static uint USizeLoc;
        internal static uint UColorLoc;

        static float mouseX, mouseY;
        static bool mouseDown;

        private static GameTicker _tick;

        public static GameTicker Tick
        {
            get { if (_tick == null) { _tick = new GameTicker(100); } return _tick; }
            set { _tick = value; }
        }

        [STAThread]
        public static void Main(string[] args)
        {
            Console.SetOut(new StreamWriter("pong_log.txt") { AutoFlush = true });
            WindowOptions options = WindowOptions.Default with { Size = new Silk.NET.Maths.Vector2D<int>(Width, Height), Title = "Pong" };
            PongWindow = Window.Create(options);
            InitScreens(PongWindow);
            PongWindow.Load +=  Pong_Load; //OnLoad;//
            PongWindow.Update += Pong_Update;
            PongWindow.Render += Pong_Render; //OnRender;// 
            PongWindow.Closing += Pong_Close;
            PongWindow.Run();
        }

        #region Initialization
        static void InitScreens(IWindow window)
        {
            mainMenuScreen = new Screen { ScreenState = UI.MenuStates.MainMenu };
            settingsScreen = new Screen { ScreenState = UI.MenuStates.SettingsMenu };

            mainMenuScreen.Buttons.Add(new UI.Button(300, 300, 200, 50, Statics.MainMenu.NewGameMenuItem, () =>
            {
                Console.WriteLine("Start Game");
                // later: switch to game state
            }));
            mainMenuScreen.Buttons.Add(new UI.Button(300, 230, 200, 50, Statics.MainMenu.SettingsMenuItem, () =>
            {
                currentScreen = settingsScreen;
            }));
            mainMenuScreen.Buttons.Add(new UI.Button(300, 160, 200, 50, Statics.MainMenu.ExitMenuItem, () =>
            {
                window.Close();
            }));

            /*
             * case MenuStates.PauseMenu:
             *      uxMainMenu.Buttons.Add(MainMenu.ResumeGameMenuItem);
             *      uxMainMenu.Buttons.Add(MainMenu.NewGameMenuItem);
             *      uxMainMenu.Buttons.Add(MainMenu.SettingsMenuItem);
             *      uxMainMenu.Buttons.Add(MainMenu.ExitGameMenuItem);
             *      break;
             * 
             */

            settingsScreen.Buttons.Add(new UI.Button(300, 300, 200, 50, Statics.MainMenu.EnableTrainingMode, () =>
            {
                Console.WriteLine("Toggle Training Mode");
                // later: toggle training mode setting
            }));
            settingsScreen.Buttons.Add(new UI.Button(300, 230, 200, 50, Statics.MainMenu.UseTrainedAIOpponent, () =>
            {
                Console.WriteLine("Toggle Trained AI Opponent");
                // later: toggle trained AI opponent setting
            }));
            settingsScreen.Buttons.Add(new UI.Button(300, 160, 200, 50, Statics.MainMenu.ApplySettingsMenuItem, () =>
            {
                Console.WriteLine("Apply Settings");
                // later: apply settings and return to main menu

                currentScreen = mainMenuScreen;
            }));

            // init with main menu
            currentScreen = mainMenuScreen;
        }

        static void HookInput(IWindow window)
        {
            var input = window.CreateInput();
            Mouse = input.Mice.FirstOrDefault();
            Keyboard = input.Keyboards.FirstOrDefault();

            if (Mouse != null)
            {
                Mouse.MouseMove += OnMouseMove;
                Mouse.MouseDown += OnMouseDown;
                Mouse.MouseUp += OnMouseUp;
            }
        }

        static unsafe void InitGLResources()
        {
            QuadVao = GLHelper.CreateQuadVao(GlContext);
            UIShaderProgram = GLHelper.CreateShaderProgram(GlContext);

            //Tell opengl how to give the data to the shaders.
            GlContext.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), null);
            GlContext.EnableVertexAttribArray(0);

            DebugGLState("After creating GL resources");

            Console.WriteLine($"Created VAO: {QuadVao}, Program: {UIShaderProgram}");

            // Verify VAO is valid
            if (QuadVao == 0)
                Console.WriteLine("ERROR: QuadVao is 0!");
            if (UIShaderProgram == 0)
                Console.WriteLine("ERROR: UIShaderProgram is 0!");

            SetUniformLocations();

            Console.WriteLine($"Uniforms - Screen: {UScreenSizeLoc}, Pos: {UPositionLoc}, Size: {USizeLoc}, Color: {UColorLoc}");
            
        }

        static void SetUniformLocations()
        {
            UScreenSizeLoc = (uint)GlContext.GetUniformLocation(UIShaderProgram, "uScreenSize");
            UPositionLoc = (uint)GlContext.GetUniformLocation(UIShaderProgram, "uPosition");
            USizeLoc = (uint)GlContext.GetUniformLocation(UIShaderProgram, "uSize");
            UColorLoc = (uint)GlContext.GetUniformLocation(UIShaderProgram, "uColor");

            Console.WriteLine($"Uniforms - Screen: {UScreenSizeLoc}, Pos: {UPositionLoc}, Size: {USizeLoc}, Color: {UColorLoc}");
        }
        #endregion

        #region Events
        static void UpdateUI()
        {
            foreach (var btn in currentScreen.Buttons)
            {
                if (btn.Contains(mouseX, mouseY))
                {
                    btn.State = mouseDown ? ButtonState.Pressed : ButtonState.Hover;
                }
                else
                {
                    btn.State = ButtonState.Normal;
                }
            }
        }

        static void DrawUI()
        {
            foreach (var btn in currentScreen.Buttons)
            {
                Console.WriteLine($"Drawing button: {btn.Text} at ({btn.X}, {btn.Y}) with state {btn.State}");
                DrawButton(btn);
            }
        }

        static void HandleClick(float mx, float my)
        {
            foreach (var btn in currentScreen.Buttons)
            {
                if (btn.Contains(mx, my))
                {
                    btn.OnClick?.Invoke();
                    break;
                }
            }
        }

        private static void OnMouseMove(IMouse mouse, Vector2 position)
        {
            mouseX = position.X;
            mouseY = PongWindow.Size.Y - position.Y; // flip Y for UI coordinates
        }

        private static void OnMouseDown(IMouse mouse, MouseButton button)
        {
            if (button == MouseButton.Left)
                mouseDown = true;
        }

        private static void OnMouseUp(IMouse mouse, MouseButton button)
        {
            if (button == MouseButton.Left)
            {
                mouseDown = false;
                HandleClick(mouseX, mouseY);
            }
        }
        #endregion

        #region Rendering
        static void DrawButton(UI.Button btn)
        {
            Console.WriteLine($"About to draw: VAO={QuadVao}, Program={UIShaderProgram}\r\nColor: ({btn.R}, {btn.G}, {btn.B}) Location: {btn.X}, {btn.Y}");
            GLHelper.DrawQuad(GlContext, UIShaderProgram, QuadVao, btn.X, btn.Y, btn.Width, btn.Height, btn.GetColor(), (Width, Height));
        }

        #endregion

        #region Window Events
        private static void Pong_Update(double obj)
        {
            UpdateUI();
        }
        private static void Pong_Render(double obj)
        {
            GlContext.ClearColor(0.15f, 0.25f, 0.30f, 1.0f);
            GlContext.Clear((uint)ClearBufferMask.ColorBufferBit);

            GlContext.BindVertexArray(QuadVao);
            GlContext.UseProgram(UIShaderProgram);

            SetUniformLocations();

            DrawUI();
        }

        private static void Pong_Load()
        {
            GlContext = GL.GetApi(PongWindow);
            Console.WriteLine($"GL Version: {GlContext.GetStringS(StringName.Version)}");

            InitGLResources();

            GlContext.Viewport(0, 0, (uint)PongWindow.Size.X, (uint)PongWindow.Size.Y);

            // Debug state
            var scissorEnabled = GlContext.IsEnabled(EnableCap.ScissorTest);
            var stencilEnabled = GlContext.IsEnabled(EnableCap.StencilTest);
            var blendEnabled = GlContext.IsEnabled(EnableCap.Blend);
            Console.WriteLine($"State - Scissor: {scissorEnabled}, Stencil: {stencilEnabled}, Blend: {blendEnabled}");


            Console.WriteLine($"Shader Program: {UIShaderProgram}");
            Console.WriteLine($"Quad VAO: {QuadVao}");
            Console.WriteLine($"Viewport: 0,0,{PongWindow.Size.X},{PongWindow.Size.Y}");

            GlContext.GetProgram(UIShaderProgram, ProgramPropertyARB.LinkStatus, out int linkStatus);
            Console.WriteLine($"Shader link status: {linkStatus}");
            if (linkStatus == 0)
            {
                string infoLog = GlContext.GetProgramInfoLog(UIShaderProgram);
                Console.WriteLine($"ERROR: {infoLog}");
            }

            HookInput(PongWindow);
        }
        #endregion

        #region Deconstruction
        static void Pong_Close()
        {
            if (Mouse != null)
            {
                Mouse.MouseMove -= OnMouseMove;
                Mouse.MouseDown -= OnMouseDown;
                Mouse.MouseUp -= OnMouseUp;
            }


        }
        #endregion

        private static void DebugGLState(string context)
        {
            GLHelper.LogGlError(GlContext, context);
        }

        #region test
        //private static GL Gl;

        //private static uint Shader;

        ////Vertex shaders are run on each vertex.
        //private static readonly string VertexShaderSource = @"
        //#version 330 core //Using version GLSL version 3.3
        //layout (location = 0) in vec4 vPos;
        
        //void main()
        //{
        //    gl_Position = vec4(vPos.x, vPos.y, vPos.z, 1.0);
        //}
        //";

        ////Fragment shaders are run on each fragment/pixel of the geometry.
        //private static readonly string FragmentShaderSource = @"
        //#version 330 core
        //out vec4 FragColor;

        //void main()
        //{
        //    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
        //}
        //";

        ////Vertex data, uploaded to the VBO.
        //private static readonly float[] Vertices =
        //{
        //    //X    Y      Z
        //     0.5f,  0.5f, 0.0f,
        //     0.5f, -0.5f, 0.0f,
        //    -0.5f, -0.5f, 0.0f,
        //    -0.5f,  0.5f, 0.5f
        //};

        ////Index data, uploaded to the EBO.
        //private static readonly uint[] Indices =
        //{
        //    0, 1, 3,
        //    1, 2, 3
        //};
        //private static unsafe void OnLoad()
        //{
        //    IInputContext input = PongWindow.CreateInput();
        //    for (int i = 0; i < input.Keyboards.Count; i++)
        //    {
        //        //input.Keyboards[i].KeyDown += KeyDown;
        //    }

        //    //Getting the opengl api for drawing to the screen.
        //    Gl = GL.GetApi(PongWindow);

        //    //Creating a vertex array.
        //    Vao = Gl.GenVertexArray();
        //    Gl.BindVertexArray(Vao);

        //    //Initializing a vertex buffer that holds the vertex data.
        //    Vbo = Gl.GenBuffer(); //Creating the buffer.
        //    Gl.BindBuffer(BufferTargetARB.ArrayBuffer, Vbo); //Binding the buffer.
        //    fixed (void* v = &Vertices[0])
        //    {
        //        Gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(Vertices.Length * sizeof(uint)), v, BufferUsageARB.StaticDraw); //Setting buffer data.
        //    }

        //    //Initializing a element buffer that holds the index data.
        //    Ebo = Gl.GenBuffer(); //Creating the buffer.
        //    Gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, Ebo); //Binding the buffer.
        //    fixed (void* i = &Indices[0])
        //    {
        //        Gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(Indices.Length * sizeof(uint)), i, BufferUsageARB.StaticDraw); //Setting buffer data.
        //    }

        //    //Creating a vertex shader.
        //    uint vertexShader = Gl.CreateShader(ShaderType.VertexShader);
        //    Gl.ShaderSource(vertexShader, VertexShaderSource);
        //    Gl.CompileShader(vertexShader);

        //    //Checking the shader for compilation errors.
        //    string infoLog = Gl.GetShaderInfoLog(vertexShader);
        //    if (!string.IsNullOrWhiteSpace(infoLog))
        //    {
        //        Console.WriteLine($"Error compiling vertex shader {infoLog}");
        //    }

        //    //Creating a fragment shader.
        //    uint fragmentShader = Gl.CreateShader(ShaderType.FragmentShader);
        //    Gl.ShaderSource(fragmentShader, FragmentShaderSource);
        //    Gl.CompileShader(fragmentShader);

        //    //Checking the shader for compilation errors.
        //    infoLog = Gl.GetShaderInfoLog(fragmentShader);
        //    if (!string.IsNullOrWhiteSpace(infoLog))
        //    {
        //        Console.WriteLine($"Error compiling fragment shader {infoLog}");
        //    }

        //    //Combining the shaders under one shader program.
        //    Shader = Gl.CreateProgram();
        //    Gl.AttachShader(Shader, vertexShader);
        //    Gl.AttachShader(Shader, fragmentShader);
        //    Gl.LinkProgram(Shader);

        //    //Checking the linking for errors.
        //    Gl.GetProgram(Shader, GLEnum.LinkStatus, out var status);
        //    if (status == 0)
        //    {
        //        Console.WriteLine($"Error linking shader {Gl.GetProgramInfoLog(Shader)}");
        //    }

        //    //Delete the no longer useful individual shaders;
        //    Gl.DetachShader(Shader, vertexShader);
        //    Gl.DetachShader(Shader, fragmentShader);
        //    Gl.DeleteShader(vertexShader);
        //    Gl.DeleteShader(fragmentShader);

        //    //Tell opengl how to give the data to the shaders.
        //    Gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), null);
        //    Gl.EnableVertexAttribArray(0);
        //}

        //private static unsafe void OnRender(double obj) //Method needs to be unsafe due to draw elements.
        //{
        //    //Clear the color channel.
        //    Gl.Clear((uint)ClearBufferMask.ColorBufferBit);

        //    //Bind the geometry and shader.
        //    Gl.BindVertexArray(Vao);
        //    Gl.UseProgram(Shader);

        //    //Draw the geometry.
        //    Gl.DrawElements(PrimitiveType.Triangles, (uint)Indices.Length, DrawElementsType.UnsignedInt, null);
        //}
        #endregion
    }
}
