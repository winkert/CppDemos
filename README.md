# CppDemos
## Purpose
Generally, I learn best through doing and teaching. To that end, this repository is a collection of OpenGl demos I will be building along the way as I learn how to code with OpenGl and advance my general knowledge of C++. This will be a combination of "vibe coding" experiments and lessons learned from documentation and existing tutorials. 
## Contents
* Main.cpp in this src/ root is the starter project. As I move forward in this learning exercise, I'll add new folders and projects.
* include/shaders/*.glsl
* * These are shader fragment files. The purpose is to keep the c++ code cleaner by moving shader logic to files
* common/
* * shader_utils - This handles getting the shader files and and creating the shaders
* * render_object - This handles rendering of each object/element being created in the scene
## Change Notes
* 9/11/2026 - Completed Cross-Platform support for current state. Tested on Mint Linux via VirtualBox
* 9/18/2026 - Started the process of abstracting and making the whole thing more Object Oritented. Added RenderObject and a (not yet working) add_element method to manage adding new triangles or objects for rendering in an efficient way.
* 9/20/2026 - Implemented add_element method to add each individual object to be rendered to a list with instructions for rendering. This finalized a process started a few days before.
* 9/21/2026 - Implemented a color array buffer to go with the vertex array buffer. Following this tutorial (https://antongerdelan.net/opengl/vertexbuffers.html) I added the color buffer. This colors each corner of the triangles to be a different color. The result is pretty, if maybe not 100% practical. 
* 9/24/2026 - I tried implementing a hot reload of the shaders following a tutorial. When I press the reload button, the graphics freeze and never recover. Something to keep hacking at.
## Notes (stream of consciousness as I go)
* I had a bit of an aha moment one night looking at (https://learnopengl.com/Getting-started/Hello-Triangle) and the solution to some of the exercises. For example, I learned how to draw multiple objects using multiple arrays etc. My thought now is I should have some kind of "RenderObject" class that contains a reference to the VAO and what kind of array it is drawing (GL_TRIANGLES, 0, 3). Then create multiples of those in an array that is looped to render in sequence.
* I implemented (with some help from Copilot) a RenderObject and it does work (with at least one object) so this is a good start
* One thing I would like to implement next is to embed the color details into the RenderObject. This would mean setting the color values in each render() call instead of one time at the top of the render method.
* I had lost the tutorial I was following (I think I was following a specific tutorial) so now I'm freelwheeling a bit. Building architecture and abstractions around this code is what I guess I do when I have lost my direction.
* A good potential enhancement with the creation of the RenderObject is that I could create a flatfile that can be used to define each triangle. The flatfile would define a triangle, the vertices, and potentially the color. Eventually I am moving the position logic out of the shader files into the c++ or some other place, and writing a flatfile to control that is a logical a process as any other. It allows changing and experimentation without recompiling.
* I'm going to take a break from this for a bit. I fell into this while working through a book on game programming in C++ and OpenGL. That book uses sprites which is a little different. 