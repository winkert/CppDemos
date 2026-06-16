//#include <GLAD/glad.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

// compile with g++ src/main.cpp -o bin/app.exe -Iinclude -Llib -lglfw3 -lglew32 -lopengl32 -lgdi32
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
GLuint vs, fs, program, attribute_coord2d;

void write_error(const char* message){
    std::cerr << message << std::endl;
}

bool init_resources(void){
    
    vs = glCreateShader(GL_VERTEX_SHADER);
    const char *vs_source =
		//"#version 100\n"  // OpenGL ES 2.0
		"#version 120\n"  // OpenGL 2.1
		"attribute vec2 coord2d;                  "
		"void main(void) {                        "
		"  gl_Position = vec4(coord2d, 0.0, 1.0); "
		"}";
    glShaderSource(vs, 1, &vs_source, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &compile_ok);
    if (!compile_ok) {
        write_error("Error in vertext shader");
        return false;
    }

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char *fs_source =
        //"#version 100\n"  // OpenGL ES 2.0
        "#version 120\n"  // OpenGL 2.1
        "void main(void) {        "
        "  gl_FragColor[0] = 0.0; "
        "  gl_FragColor[1] = 0.0; "
        "  gl_FragColor[2] = 1.0; "
        "}";
    glShaderSource(fs, 1, &fs_source, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &compile_ok);
    if (!compile_ok) {
        write_error("Error in fragment shader");
        return false;
    }

    program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
	if (!link_ok) {
		write_error("Error in glLinkProgram");
		return false;
	}

    const char* attribute_name = "coord2d";
	attribute_coord2d = glGetAttribLocation(program, attribute_name);
	if (attribute_coord2d == -1) {
		write_error("Could not bind attribute");
		return false;
	}

    return true;
}

void render(GLFWwindow* window){
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);
	glEnableVertexAttribArray(attribute_coord2d);
    GLfloat triangle_vertices[] = {
    0.0,  0.8,
    -0.8, -0.8,
    0.8, -0.8,
    };

    /* Describe our vertices array to OpenGL (it can't guess its format automatically) */
    glVertexAttribPointer(
		attribute_coord2d, // attribute
		2,                 // number of elements per vertex, here (x,y)
		GL_FLOAT,          // the type of each element
		GL_FALSE,          // take our values as-is
		0,                 // no extra data between each position
		triangle_vertices  // pointer to the C array
						  );
	
	/* Push each element in buffer_vertices to the vertex shader */
	glDrawArrays(GL_TRIANGLES, 0, 3);
	
	glDisableVertexAttribArray(attribute_coord2d);

}

void cleanup_resources(void){

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

    // 2. Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Window", nullptr, nullptr);
    if (!window) {
        std::cout << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW\n";
        return -1;
    }

    if(!init_resources()){
        std::cout << "Failed to initialize resources\n";
        return -1;
    }

    mainloop(window);


    // 5. Cleanup
    cleanup_resources();
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
