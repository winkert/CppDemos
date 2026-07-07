#version 460 core
in vec3 vertPos;
out vec4 FragColor;
uniform float r;
uniform float g;
uniform float b;
uniform float alpha;
void main(void) {
    
    FragColor.r = r;
    FragColor.g = g;
    FragColor.b = b;
    FragColor.a = alpha;

    FragColor = vec4(vertPos, 1.0);

}