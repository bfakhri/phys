#include "cube.h"
#include "math.h"


///////////////
// Constructors
///////////////

Cube::Cube()
{
	sideLength = 1;
}

Cube::Cube(double l, double sMass, cart tPos, cart tVel, cart rPos, cart rVel): Shape(sMass, tPos, tVel, rPos, rVel)
{
	sideLength = l; 
}


/////////////////
// Drawing Functs
/////////////////

void Cube::drawShape()
{
	//glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	// This is just to make the rest more concise
	double off = sideLength/2;
	// Draws a cube
	//	Front
	glBegin(GL_QUADS);
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f( off,  off,  off); 
	glVertex3f(-off,  off,  off); 
	glVertex3f(-off, -off,  off); 
	glVertex3f( off, -off,  off); 
	//	Right
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f( off,  off, -off); 
	glVertex3f( off,  off,  off); 
	glVertex3f( off, -off,  off); 
	glVertex3f( off, -off, -off); 
	//	Back	
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f( off, -off, -off); 
	glVertex3f(-off, -off, -off); 
	glVertex3f(-off,  off, -off); 
	glVertex3f( off,  off, -off); 
	//	Top
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f( off,  off, -off); 
	glVertex3f(-off,  off, -off); 
	glVertex3f(-off,  off,  off); 
	glVertex3f( off,  off,  off); 
	//	Left
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(-off,  off,  off); 
	glVertex3f(-off,  off, -off); 
	glVertex3f(-off, -off, -off); 
	glVertex3f(-off, -off,  off); 
	//	Bottom
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f( off, -off,  off); 
	glVertex3f(-off, -off,  off); 
	glVertex3f(-off, -off, -off); 
	glVertex3f( off, -off, -off); 

	glEnd();
}


/////////////////
// Physics Functs
/////////////////

double Cube::volume()
{
	return sideLength*sideLength*sideLength;
}

cart Cube::momentCM()
{
	cart mmnt = {	mass*sideLength*sideLength/6,
					mass*sideLength*sideLength/6,
					mass*sideLength*sideLength/6};
	return mmnt;
}

double Cube::density()
{
	return mass/volume();
}

double Cube::boundingSphere()
{
	return sqrt(2*sideLength*sideLength);
}

// For a Cube this is the same as boundingCube
// For a cube this would not be the same
double Cube::boundingBox()
{
	return sideLength;
}


