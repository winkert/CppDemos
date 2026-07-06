using Silk.NET.Maths;
using Silk.NET.OpenGL;
using System;
using System.Runtime.CompilerServices;

namespace TRW.Games.Pong
{
    internal static class GLHelper
    {
        internal const string uiVertexShaderSrc = @"
#version 460 core
in vec2 aPos;
uniform vec2 uScreenSize;
uniform vec2 uPosition;
uniform vec2 uSize;
void main()
{
    vec2 pixelPos = uPosition + aPos * uSize;
    float x = (pixelPos.x / uScreenSize.x) * 2.0 - 1.0;
    float y = 1.0 - (pixelPos.y / uScreenSize.y) * 2.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
";

        internal const string uiFragmentShaderSrc = @"
#version 460 core
out vec4 FragColor;
uniform vec3 uColor;
void main()
{
    FragColor = vec4(uColor, 1.0);
}
";

        internal static uint CreateShaderProgram(GL gl)
        {
            uint programs = 0;

            uint vertexShader = gl.CreateShader(ShaderType.VertexShader);
            gl.ShaderSource(vertexShader, uiVertexShaderSrc);
            gl.CompileShader(vertexShader);

            gl.GetShader(vertexShader, ShaderParameterName.CompileStatus, out int vertexCompileStatus);
            if (vertexCompileStatus == 0)
            {
                string infoLog = gl.GetShaderInfoLog(vertexShader);
                Console.WriteLine($"Vertex shader compilation failed: {infoLog}");
            }
            else
            {
                Console.WriteLine("Vertex shader compiled successfully");
            }

            uint fragmentShader = gl.CreateShader(ShaderType.FragmentShader);
            gl.ShaderSource(fragmentShader, uiFragmentShaderSrc);
            gl.CompileShader(fragmentShader);

            gl.GetShader(fragmentShader, ShaderParameterName.CompileStatus, out int fragmentCompileStatus);
            if (fragmentCompileStatus == 0)
            {
                string infoLog = gl.GetShaderInfoLog(fragmentShader);
                Console.WriteLine($"Fragment shader compilation failed: {infoLog}");
            }

            programs = gl.CreateProgram();
            gl.AttachShader(programs, vertexShader);
            gl.AttachShader(programs, fragmentShader);
            gl.LinkProgram(programs);

            gl.GetProgram(programs, ProgramPropertyARB.LinkStatus, out int linkStatus);
            if (linkStatus == 0)
            {
                string infoLog = gl.GetProgramInfoLog(programs);
                Console.WriteLine($"Shader program linking failed: {infoLog}");
            }

            // Check for attribute location
            int aPosLoc = gl.GetAttribLocation(programs, "aPos");
            Console.WriteLine($"Shader attribute 'aPos' location: {aPosLoc}");

            gl.DetachShader(programs, vertexShader);
            gl.DetachShader(programs, fragmentShader);
            gl.DeleteShader(vertexShader);
            gl.DeleteShader(fragmentShader);

            return programs;
        }

        internal static unsafe uint CreateQuadVao(GL gl)
        {
            float[] quadVertices =
            {
                //X    Y      Z
                 0.5f,  0.5f, 0.0f,
                 0.5f, -0.5f, 0.0f,
                -0.5f, -0.5f, 0.0f,
                -0.5f,  0.5f, 0.5f
            };
            uint[] quadIndices = { 
                0, 1, 3,
                1, 2, 3
            };

            //Creating a vertex array.
            uint vao = gl.GenVertexArray();
            gl.BindVertexArray(vao);

            //Initializing a vertex buffer that holds the vertex data.
            uint vbo = gl.GenBuffer();
            gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
            fixed (void* v = &quadVertices[0])
            {
                gl.BufferData(BufferTargetARB.ArrayBuffer, (nuint)(quadVertices.Length * sizeof(uint)), v, BufferUsageARB.StaticDraw); //Setting buffer data.
            }

            uint ebo = gl.GenBuffer();
            gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
            fixed (void* i = &quadIndices[0])
            {
                gl.BufferData(BufferTargetARB.ElementArrayBuffer, (nuint)(quadIndices.Length * sizeof(uint)), i, BufferUsageARB.StaticDraw); //Setting buffer data.
            }

            return vao;
        }

        internal static void DrawQuad(GL gl, uint shaderProgram, uint vao, float x, float y, float width, float height, (float R, float G, float B) color, (float Width, float Height) windowSize)
        {
            gl.UseProgram(shaderProgram);
            gl.BindVertexArray(vao);

            int screenSizeLoc = gl.GetUniformLocation(shaderProgram, "uScreenSize");
            int positionLoc = gl.GetUniformLocation(shaderProgram, "uPosition");
            int sizeLoc = gl.GetUniformLocation(shaderProgram, "uSize");
            int colorLoc = gl.GetUniformLocation(shaderProgram, "uColor");
            gl.Uniform2(screenSizeLoc, windowSize.Width, windowSize.Height);
            gl.Uniform2(positionLoc, x, y);
            gl.Uniform2(sizeLoc, width, height);
            gl.Uniform3(colorLoc, color.R, color.G, color.B);

            LogGlInfo(gl, shaderProgram, "DrawQuad", (uint)screenSizeLoc, (uint)positionLoc, (uint)sizeLoc, (uint)colorLoc);

            gl.DrawElements(PrimitiveType.Triangles, 6, DrawElementsType.UnsignedInt, 0);
            LogGlError(gl, "DrawQuad");
        }

        internal static void LogGlError(GL gl, string context)
        {
            var error = gl.GetError();
            if (error != GLEnum.NoError)
            {
                Console.WriteLine($"OpenGL error in {context}: {error}");
            }
        }

        internal static void LogGlInfo(GL gl, uint shaderProgram, string context, uint screenSizeLoc, uint positionLoc, uint sizeLoc, uint colorLoc)
        {
            gl.GetProgramInfoLog(shaderProgram, out string programInfoLog);
            gl.GetShaderInfoLog(shaderProgram, out string shaderInfoLog);
            Console.WriteLine($"OpenGL Info Log for {context}:");
            Console.WriteLine($"\tProgram Info: {programInfoLog}");
            Console.WriteLine($"\tShader Info: {shaderInfoLog}");

        }
    }
}
