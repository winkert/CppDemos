#include <GL/glew.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>

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
