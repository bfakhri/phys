// This file defines functions for the rendering and physics engine
// along with important parameters that affect the behavior of both

#ifndef ENGINE_H 
#define ENGINE_H 

#include "mather.h"
#include <vector>
#include "shape.h"

///////////
// Graphics
///////////

// Framerate of the renderer
#define G_FPS 30

// For drawing 3D objects
#define DEF_SLICES 50
#define DEF_STACKS 50


//////////
// Physics
//////////

// Default period for each step (seconds)
#define SIM_T 0.01

// Global variables
const cart physOrigin = {0, 0, -20};			// Origin of the physics sim
const cart physBoundaryMax = {10, 10, 10};		// Maximum coordinates of physics sim
const cart physBoundaryMin = {-10, -10, -10};	// Minimum coordinates of physics sim

void physicsThread(std::vector<Shape*> v);


#endif
