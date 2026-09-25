#include <GL/glew.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#endif
using namespace std;

char* file_read(const char* filename)
{
	if(!filesystem::exists(filename)) {
		fprintf(stderr, "File not found: %s\n", filename);
		return NULL;
	}

	cout << "Opening File: " << filename << endl;

  	FILE* in = fopen(filename, "rb");
  	if (in == NULL) return NULL;

  	int res_size = BUFSIZ;
  	char* res = (char*)malloc(res_size);
  	int nb_read_total = 0;

  	while (!feof(in) && !ferror(in)) {
    	if (nb_read_total + BUFSIZ > res_size) {
      	if (res_size > 10*1024*1024) break;
      	res_size = res_size * 2;
      	res = (char*)realloc(res, res_size);
    	}
    char* p_res = res + nb_read_total;
    nb_read_total += fread(p_res, 1, BUFSIZ, in);
 	}
  
  	fclose(in);
  	res = (char*)realloc(res, nb_read_total + 1);
  	res[nb_read_total] = '\0';
  	return res;
}

void print_log(GLuint object) {
	GLint log_length = 0;
	if (glIsShader(object)) {
		glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length);
        cerr << "Error in shader" << endl;
	} else if (glIsProgram(object)) {
		glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length);
        cerr << "Error in program" << endl;
	} else {
		cerr << "printlog: Not a shader or a program" << endl;
		return;
	}

	char* log = (char*)malloc(log_length);
	
	if (glIsShader(object))
		glGetShaderInfoLog(object, log_length, NULL, log);
	else if (glIsProgram(object))
		glGetProgramInfoLog(object, log_length, NULL, log);
	
	cerr << log;
	free(log);
}

bool get_shader_paths(std::filesystem::path& vertex_shader_path, std::filesystem::path& fragment_shader_path) {	
	// this is default - assume current path is the bin directory
	std::filesystem::path shader_dir = std::filesystem::current_path() / "shaders";

	if(!std::filesystem::exists(shader_dir)) {
		try{
			// need to determine OS and then set the shader_dir accordingly
			#ifdef _WIN32
			wchar_t this_process_path[MAX_PATH];
			
			GetModuleFileNameW(NULL, this_process_path, sizeof(this_process_path) / sizeof(wchar_t));
			std::wcout << L"Unicode path of this app: " << this_process_path << std::endl;
			shader_dir = std::filesystem::path(this_process_path).parent_path() / "shaders";
			#elif __linux__
			// try to get symlink to current process
			std::filesystem::path exe_path = std::filesystem::read_symlink("/proc/self/exe");
			shader_dir = exe_path.parent_path() / "shaders";
			#else
			std::cerr << "Unsupported OS" << std::endl;
			return false;
			#endif
		}
		catch (const std::exception& e) {
			std::cerr << "Error determining shader directory: " << e.what() << std::endl;
			return false;
		}

		// one final check
		if(!std::filesystem::exists(shader_dir)) {
			std::cerr << "Shader directory not found: " << shader_dir << std::endl;
			return false;
		}
	}

	vertex_shader_path = shader_dir / "vertex.glsl";
	fragment_shader_path = shader_dir / "fragment.glsl";
	return true;
}
