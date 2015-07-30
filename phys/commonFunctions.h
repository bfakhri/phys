// Here are defined common helper functions (mostly for main.cpp)
// that will help make the rest of the files more concise/readable
#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include "shape.h"
#include "sphere.h"
#include <vector>

// Make random shape with param constraints
Shape* randomShape();

// Make random shape with param constraints
Shape* randomShape(double radMin, double radMax, double minMass, double maxMass, cart tMaxPos,cart tMaxVel, cart rMaxPos, cart rMaxVel);

// Maybe makes this flexible - can take in shapes from files or random shapes
// depending on the inputs to the function
void populateShapeVector(std::vector<Shape*> v);

#endif
