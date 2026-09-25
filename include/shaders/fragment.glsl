#version 410 core
in vec3 color;
out vec4 FragColor;
uniform float r;
uniform float g;
uniform float b;
uniform float alpha;
void main(void) {
    if(length(color) > 0.001 ){
        FragColor = vec4(color, 1.0);
    }
    else{
        FragColor.r = r;
        FragColor.g = g;
        FragColor.b = b;
        FragColor.a = alpha;
    }
}