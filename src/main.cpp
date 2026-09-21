//#include <GLAD/glad.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <filesystem>
#include <math.h>
#include "shader_utils.h"
#include "render_object.h"

// compile with 
// WINDOWS:
//  g++ src/*.cpp, src/common/*.cpp -o bin/app.exe -Iinclude -Isrc/common -Llib -lglfw3 -lglew32 -lopengl32 -lgdi32
// LINUX:
//  g++ src/*.cpp src/common/*.cpp -o bin/app -Iinclude -Isrc/common -lGL -lglfw -lGLEW 
// lib and include files from these sources:
// - GLEW:
//   - https://glew.sourceforge.net/
//   - Included is 2.3.1-win32.zip
// - GLFW:
//   - https://www.glfw.org/download.html
//  - Included is glfw-3.4.bin.WIN64.zip
//  - Locally, I'm using Mingw-w64 to compile so I used that version of this dll
// - GLAD:
//   - https://glad.dav1d.de/
//   - I generated a custom loader for OpenGL 4.6 core profile and downloaded
//   - Whatever I generated does not compile locally, so I am excluding the include but leaving it here for reference
const unsigned int RENDEROBJECT_COUNT = 3;
const unsigned int VERTEX_LOCATION = 0;
const unsigned int COLOR_LOCATION = 1;

GLint compile_ok = GL_FALSE, link_ok = GL_FALSE;
GLuint vs, fs, program, u_r, u_g, u_b, u_alpha, u_time;
GLuint vertex_arrays[RENDEROBJECT_COUNT], vertex_buffers[RENDEROBJECT_COUNT], color_buffers[RENDEROBJECT_COUNT];
RenderObject objects[RENDEROBJECT_COUNT];

void write_error(const char* message){
    std::cerr << message << std::endl;
}

template<typename... T>
void write_log(const T&... messages){
    (std::cout << ... << messages) << std::endl;
}

bool init_resources(void){
    std::filesystem::path vertex_shader_path, fragment_shader_path;
    if(!get_shader_paths(vertex_shader_path, fragment_shader_path)) {
        return false;
    }
    
    // log renderer info
    const char* vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    write_log("OpenGL vendor: ", vendor);
    write_log("Renderer: ", renderer);
    write_log("OpenGL version supported: ", version);

    GLchar infoLog[1024];
    vs = glCreateShader(GL_VERTEX_SHADER);
    const char *vs_source = file_read(vertex_shader_path.string().c_str());
    glShaderSource(vs, 1, &vs_source, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &compile_ok);
    if (!compile_ok) {
        glGetShaderInfoLog(vs, sizeof(infoLog), nullptr, infoLog);
        write_error("Error in vertext shader");
        write_error(infoLog);
        return false;
    }

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char *fs_source = file_read(fragment_shader_path.string().c_str());
    glShaderSource(fs, 1, &fs_source, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &compile_ok);
    if (!compile_ok) {
        glGetShaderInfoLog(fs, sizeof(infoLog), nullptr, infoLog);
        write_error("Error in fragment shader");
        write_error(infoLog);
        return false;
    }

    program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
    
    glBindAttribLocation(program, VERTEX_LOCATION, "vertex_position");
    glBindAttribLocation(program, COLOR_LOCATION, "vertex_color");

	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
	if (!link_ok) {
        glGetProgramInfoLog(program, sizeof(infoLog), nullptr, infoLog);
		write_error("Error in glLinkProgram");
        write_error(infoLog);
		return false;
	}

	u_r = glGetUniformLocation(program, "r");
	u_g = glGetUniformLocation(program, "g");
	u_b = glGetUniformLocation(program, "b");
	u_alpha = glGetUniformLocation(program, "alpha");
    u_time = glGetUniformLocation(program, "time");

    glDeleteShader(vs);
    glDeleteShader(fs);

    write_log("Shader program compiled and linked successfully.");
    write_log("shader program ID: ", program);
    write_log("Uniform 'r' location: ", u_r);
    write_log("Uniform 'g' location: ", u_g);
    write_log("Uniform 'b' location: ", u_b);
    write_log("Uniform 'alpha' location: ", u_alpha);
    write_log("Uniform 'time' location: ", u_time);

    return true;
}

void add_element(GLfloat* vertices, GLfloat* colors, unsigned int index, unsigned int vertices_count = 3, unsigned int dimensions = 2){
    // previous iterations of this code used sizeof(vertices) to determine the number of vertices, but that is incorrect.
    // vertices is a pointer to an array - NOT the array itself. 
    // sizeof(vertices) will return the size of the pointer, not the array. So we need to pass in the number of vertices as a parameter.
    glBindVertexArray(vertex_arrays[index]);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffers[index]);
    glBufferData(GL_ARRAY_BUFFER, vertices_count * dimensions * sizeof(GLfloat), vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(
        VERTEX_LOCATION, // attribute
        dimensions,                 // number of elements per vertex, here (x,y)
        GL_FLOAT,          // the type of each element
        GL_FALSE,          // take our values as-is
        0,                  // no space between data
        nullptr  // pointer to the C array
    );

    glBindBuffer(GL_ARRAY_BUFFER, color_buffers[index]);
    glBufferData(GL_ARRAY_BUFFER, vertices_count * 4 * sizeof(GLfloat), colors, GL_STATIC_DRAW);
    glVertexAttribPointer(
        COLOR_LOCATION, // attribute
        4,                     // number of elements per vertex, here (r,g,b,a)
        GL_FLOAT,          // the type of each element
        GL_FALSE,          // take our values as-is
        0,                  // no space between data
        nullptr  // pointer to the C array
    );

    objects[index] = RenderObject();
    objects[index].init({vertex_arrays[index], vertex_buffers[index], program, GL_TRIANGLES, 0, vertices_count, false, false, false});
    
    glEnableVertexAttribArray(VERTEX_LOCATION);
    glEnableVertexAttribArray(COLOR_LOCATION);
}

void init_buffers(void){
    GLfloat vertices[] = {
    0.0,  0.8,
    -0.8, -0.8,
    0.8, -0.8,
    };
   float firstTriangle[] = {
        -0.9f, -0.5f, 0.0f,  // left 
        -0.0f, -0.5f, 0.0f,  // right
        -0.45f, 0.5f, 0.0f,  // top 
    };
    float secondTriangle[] = {
        0.0f, -0.5f, 0.0f,  // left
        0.9f, -0.5f, 0.0f,  // right
        0.45f, 0.5f, 0.0f   // top 
    };

    float colors[] = {
        1.0f, 0.0f, 0.0f, 1.0f, // red
        0.0f, 1.0f, 0.0f, 1.0f, // green
        0.0f, 0.0f, 1.0f, 1.0f  // blue
    };
    glGenVertexArrays(RENDEROBJECT_COUNT, vertex_arrays);
    glGenBuffers(RENDEROBJECT_COUNT, vertex_buffers);
    glGenBuffers(RENDEROBJECT_COUNT, color_buffers);

    add_element(vertices, colors, 0);
    add_element(firstTriangle, colors, 1, 3, 3);
    add_element(secondTriangle, colors, 2, 3, 3);
    
}

void render(GLFWwindow* window){
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(program);
    double curr_s = glfwGetTime();

    objects[0].color[0] = 0.6f;
    objects[0].color[1] = 0.8f;
    objects[0].color[2] = 0.2f;
    objects[0].color[3] = 0.5f;

    // make the triangle move
    glUniform1f(u_time, (float)curr_s);

    for (const auto& obj : objects) {
        glUniform1f(u_r, obj.color[0]);
        glUniform1f(u_g, obj.color[1]);
        glUniform1f(u_b, obj.color[2]);
        glUniform1f(u_alpha, obj.color[3]);
        obj.render();
    }
	
}

void cleanup_resources(void){
    glDisableVertexAttribArray(VERTEX_LOCATION);
    glDeleteVertexArrays(RENDEROBJECT_COUNT, vertex_arrays);
    glDeleteBuffers(RENDEROBJECT_COUNT, vertex_buffers);
    glDeleteProgram(program);

}

void mainloop(GLFWwindow* window){
    while (!glfwWindowShouldClose(window)) {
        // --- Input ---
        glfwPollEvents();

        // handle input (e.g., close window on ESC key press)
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
           glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // --- Rendering ---
        render(window);

        // --- Swap buffers ---
        glfwSwapBuffers(window);
    }
}

int main() {
    int result = 0;
    // 1. Initialize GLFW
    if (!glfwInit()) {
        write_error("Failed to initialize GLFW\n");
        return -1;
    }

      // Request an OpenGL 4.1, core, context from GLFW.
    glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
    glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 1 );
    glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE );
    glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

    // 2. Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Window", nullptr, nullptr);
    if (!window) {
        write_error("Failed to create window\n");
        glfwTerminate();
        return -1;
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        write_error("Failed to initialize GLEW\n");
        return -1;
    }

    if(!init_resources()){
        write_error("Failed to initialize resources\n");
        return -1;
    }

    init_buffers();

    mainloop(window);


    // 5. Cleanup
    cleanup_resources();
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
