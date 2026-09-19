// src/common/render_object.cpp
#include "render_object.h"
#include <iostream>

RenderObject::RenderObject() = default;

RenderObject::~RenderObject() {
    cleanup();
}

bool RenderObject::init(const InitParams &p) {
    // Caller must ensure the appropriate GL context is current before calling.
    if (p.count == 0) {
        std::cerr << "RenderObject::init: vertex count is zero\n";
        return false;
    }
    if (p.vao == 0 || p.vbo == 0 || p.program == 0) {
        std::cerr << "RenderObject::init: invalid GL handles\n";
        return false;
    }

    // Optional: verify handles with GL. Skip these checks if context may be different.
    if (!glIsVertexArray(p.vao) || !glIsBuffer(p.vbo) || !glIsProgram(p.program)) {
        std::cerr << "RenderObject::init: glIs* checks failed\n";
        return false;
    }

    // If re-initializing, release previous owned resources first.
    cleanup();

    vertex_array = p.vao;
    vertex_buffer = p.vbo;
    program = p.program;
    render_mode = p.mode;
    first_vertex = p.first;
    vertex_count = p.count;

    owns_vao = p.ownsVao;
    owns_vbo = p.ownsVbo;
    owns_program = p.ownsProgram;

    return true;
}

void RenderObject::render() const {
    if (!is_valid()) return;
    glUseProgram(program);
    glBindVertexArray(vertex_array);
    glDrawArrays(render_mode, static_cast<GLint>(first_vertex), static_cast<GLsizei>(vertex_count));
    glBindVertexArray(0);
    glUseProgram(0);
}

void RenderObject::cleanup() {
    if (owns_vbo && vertex_buffer != 0) {
        glDeleteBuffers(1, &vertex_buffer);
        vertex_buffer = 0;
    }
    if (owns_vao && vertex_array != 0) {
        glDeleteVertexArrays(1, &vertex_array);
        vertex_array = 0;
    }
    if (owns_program && program != 0) {
        glDeleteProgram(program);
        program = 0;
    }
    vertex_count = 0;
    first_vertex = 0;
    owns_vbo = owns_vao = owns_program = false;
}