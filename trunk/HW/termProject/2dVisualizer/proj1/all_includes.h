// This file includes all header files to ensure glut.h is included last
// Necessary because of an exit() function redefinition

#include <string>		// Used for drawing scoreboard/timer
#include <iostream>		// For debugging purposes 
#include <vector>		// Vectors used to house drawable objects
#include <time.h>		// Used for tracking time 
#include <math.h>		// For trig and other math functions

#include "ShapeObjects.h"	// Classes of shapes
#include "shapes.h"			// Defines functions for drawing shapes
#include "globals.h"		// Global variables 
#include "drawShapes.h"		// Functions where the breakout room is initialized and drawn
#include "drawScore.h"		// Functions where scoreboard is initialized and drawn

#include "glut.h"		// For openGL 