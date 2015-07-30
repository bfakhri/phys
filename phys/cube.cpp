#include "Cube.h"
#include "math.h"

// Default
Cube::Cube()
{
	sideLength = 1;
}

// Calls parent non-default constructor and passes args to it
Cube::Cube(double l, double sMass, cart tPos, cart tVel, cart rPos, cart rVel): Shape(sMass, tPos, tVel, rPos, rVel)
{
	sideLength = l; 
}

double Cube::volume()
{
	return sideLength*sideLength*sideLength;
}

cart Cube::moment()
{
	cart mmnt = {	mass*sideLength*sideLength/6,
			mass*sideLength*sideLength/6,
			mass*sideLength*sideLength/6}
	return cart;
}

double Cube::density()
{
	return mass/volume();
}

double Cube::boundingCube()
{
	return sideLength;
}

// For a Cube this is the same as boundingCube
// For a cube this would not be the same
double Cube::boundingBox()
{
	return sideLength;
}
