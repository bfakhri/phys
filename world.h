// This file describes functions that help build the world
// Things such as walls/wrapping/universal gravity should be defined here

#ifndef WORLD_H
#define WORLD_H

#include "mather.h"
#include "shape.h"
#include <vector>
#include <GL/glut.h>	// For draw() function
#include <GL/gl.h>		// For draw() function

// Draw boundaries - draws world boundaries
// Only draws boundaries for those with nonzero components
// ***Make this more general to allow different drawing types***
// By default front boundary will NOT draw so we can see the scene
void drawBoundaries(cart origin, cart min, cart max);


// Enforce boundaries (act as walls of the scene)
void enforceBoundaries(std::vector<Shape*> s, cart min, cart max);

#endif
