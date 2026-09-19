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

GLint compile_ok = GL_FALSE, link_ok = GL_FALSE;
GLuint vs, fs, program, attribute_coord2d, u_r, u_g, u_b, u_alpha, u_time;
GLuint vertex_arrays[1], vertex_buffers[1];
RenderObject objects[1];

void write_error(const char* message){
    std::cerr << message << std::endl;
}

bool init_resources(void){
    std::filesystem::path vertex_shader_path, fragment_shader_path;
    if(!get_shader_paths(vertex_shader_path, fragment_shader_path)) {
        return false;
    }
    
    // log renderer info
    const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString(GL_VERSION); // version as a string
    std::cout << "Renderer: " << renderer << std::endl;
    std::cout << "OpenGL version supported: " << version << std::endl;

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
    attribute_coord2d = glGetAttribLocation(program, "coord2d");
    u_time = glGetUniformLocation(program, "time");

    glDeleteShader(vs);
    glDeleteShader(fs);

    return true;
}

void add_element(GLfloat* vertices, unsigned int index, unsigned int vertices_count = 3, unsigned int dimensions = 2){
    // for some reason using this does not work
    glBindVertexArray(vertex_arrays[index]);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffers[index]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(
        attribute_coord2d, // attribute
        dimensions,                 // number of elements per vertex, here (x,y)
        GL_FLOAT,          // the type of each element
        GL_FALSE,          // take our values as-is
        0,                  // no space between data
        NULL  // pointer to the C array
    );

    objects[index] = RenderObject();
    objects[index].init({vertex_arrays[index], vertex_buffers[index], program, GL_TRIANGLES, 0, vertices_count, false, false, false});
    
    glEnableVertexAttribArray(attribute_coord2d);
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

    glGenVertexArrays(1, vertex_arrays);
    glGenBuffers(1, vertex_buffers);

    //add_element(vertices, 0);
    //add_element(firstTriangle, 1, 3);
    //add_element(secondTriangle, 2, 3);

    glBindVertexArray(vertex_arrays[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(
        attribute_coord2d, // attribute
        2,                 // number of elements per vertex, here (x,y)
        GL_FLOAT,          // the type of each element
        GL_FALSE,          // take our values as-is
        0,                  // no space between data
        NULL  // pointer to the C array
    );
    
    objects[0] = RenderObject();
    objects[0].init({vertex_arrays[0], vertex_buffers[0], program, GL_TRIANGLES, 0, 3, false, false, false});
    
    glEnableVertexAttribArray(attribute_coord2d);

}

void render(GLFWwindow* window){
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);
    double curr_s = glfwGetTime();
    //std::cout << "Current time (glfwGetTime): " << curr_s << std::endl;
    
	glUniform1f(u_r, 0.6f);
    glUniform1f(u_g, 0.4f);
    glUniform1f(u_b, 0.2f);
    glUniform1f(u_alpha, sin(curr_s));

    // make the triangle move
    glUniform1f(u_time, (float)curr_s);

    for (const auto& obj : objects) {
        obj.render();
    }
	
}

void cleanup_resources(void){
    glDisableVertexAttribArray(attribute_coord2d);
    glDeleteVertexArrays(1, vertex_arrays);
    glDeleteBuffers(1, vertex_buffers);
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
        std::cout << "Failed to initialize GLFW\n";
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
