#version 460 core
in vec2 coord2d;
uniform float time;
out vec3 vertPos;
void main(void) {
    vec3 pos = vec3(coord2d, 0.0);
    pos.y += sin(time);
    vertPos = pos;
    gl_Position = vec4(pos, 1.0);
}   