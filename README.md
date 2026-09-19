# CppDemos
## Purpose
Generally, I learn best through doing and teaching. To that end, this repository is a collection of OpenGl demos I will be building along the way as I learn how to code with OpenGl and advance my general knowledge of C++. This will be a combination of "vibe coding" experiments and lessons learned from documentation and existing tutorials. 
## Contents
* Main.cpp in this src/ root is the starter project. As I move forward in this learning exercise, I'll add new folders and projects.
* include/shaders/*.glsl
* * These are shader fragment files. The purpose is to keep the c++ code cleaner by moving shader logic to files
* common/
** shader_utils - This handles getting the shader files and and creating the shaders
** render_object - This handles rendering of each object/element being created in the scene

## Notes
* I had a bit of an aha moment one night looking at (https://learnopengl.com/Getting-started/Hello-Triangle) and the solution to some of the exercises. For example, I learned how to draw multiple objects using multiple arrays etc. My thought now is I should have some kind of "RenderObject" class that contains a reference to the VAO and what kind of array it is drawing (GL_TRIANGLES, 0, 3). Then create multiples of those in an array that is looped to render in sequence.
* I implemented (with some help from Copilot) a RenderObject and it does work (with at least one object) so this is a good start