#version 410 core
layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_color;
uniform float time;
out vec3 color;
void main(void) {
    vec3 pos = vertex_position;
    pos.y += sin(time);
    color = vertex_color;
    gl_Position = vec4(pos, 1.0);
}   