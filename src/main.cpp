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

int main() {
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

    // 3. Initialize GLEW (must happen AFTER context creation)
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW\n";
        return -1;
    }

    // 4. Main loop
    while (!glfwWindowShouldClose(window)) {
        // --- Input ---
        glfwPollEvents();

        // --- Rendering ---
        glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // --- Swap buffers ---
        glfwSwapBuffers(window);
    }

    // 5. Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
