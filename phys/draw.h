#ifndef DRAWSHAPES_H
#define DRAWSHAPES_H

#include "mather.h"

// Maybe add versions of this depending on options like color etc
// Draws shape depending on position, volume (more important than mass), density maybe?, and radial orientation 
void drawShape(cart position, double volume); // NOT FINISHED SIGNATURE

// Function to draw all of the shapesin the shape vector
void drawAllShapes(vector<Shape> v);
#endif 
