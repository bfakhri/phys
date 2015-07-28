// Contains references to all of the global variables and constants etc

#ifndef ALL_INCLUDES_H
#define ALL_INCLUDES_H

// All globally necessary libraries
//#include "shape.h" 

// Includes for OpenGL
#include <GL/gl.h>
#include <GL/glut.h>

using namespace std;

// Defines the cartesian structure
typedef struct cart{
	double x; 
	double y;
	double z; 
}cart;


// Vector containing all "normal" shapes
// Declared in shapes.h file
//extern vector<Shape> allShapes; 

#endif
