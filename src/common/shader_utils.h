#ifndef _SHADER_UTILS_H
#define _SHADER_UTILS_H
#include <GL/glew.h>
#include <filesystem>
#include <iostream>

extern char* file_read(const char* filename);
extern void print_log(GLuint object);
bool get_shader_paths(std::filesystem::path& vertex_shader_path, std::filesystem::path& fragment_shader_path);

#endif