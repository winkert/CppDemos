// src/common/render_object.h
#ifndef RENDER_OBJECT_H
#define RENDER_OBJECT_H

#include <GL/glew.h>

class RenderObject {
public:
    struct InitParams {
        GLuint vao = 0;
        GLuint vbo = 0;
        GLuint program = 0;
        GLenum mode = GL_TRIANGLES;
        unsigned first = 0;
        unsigned count = 3;
        bool ownsVao = false;
        bool ownsVbo = false;
        bool ownsProgram = false;
    };
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    RenderObject();
    ~RenderObject();

    bool init(const InitParams &p);
    void render() const;
    void cleanup();

private:
    GLuint vertex_array = 0;
    GLuint vertex_buffer = 0;
    GLuint program = 0;
    GLenum render_mode = GL_TRIANGLES;
    unsigned int first_vertex = 0;
    unsigned int vertex_count = 0;

    bool owns_vao = false;
    bool owns_vbo = false;
    bool owns_program = false;

    bool is_valid() const { return vertex_array != 0 && vertex_buffer != 0 && program != 0 && vertex_count > 0; }
};

#endif // RENDER_OBJECT_H