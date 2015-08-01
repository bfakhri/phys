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
	// This is just to make the rest more concise
	double off = sideLength/2;
	cart p = t_position; 
	// Draws a cube
	//	Front
	glBegin(GL_QUADS);
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(p.x+off, p.y+off, p.z+off); 
	glVertex3f(p.x-off, p.y+off, p.z+off); 
	glVertex3f(p.x-off, p.y-off, p.z+off); 
	glVertex3f(p.x+off, p.y-off, p.z+off); 
	//	Right
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(p.x+off, p.y+off, p.z+off); 
	glVertex3f(p.x+off, p.y-off, p.z+off); 
	glVertex3f(p.x+off, p.y+off, p.z-off); 
	glVertex3f(p.x+off, p.y-off, p.z-off); 
	//	Back	
	glColor3f(0.0, 0.0, 1.0);
	glVertex3f(p.x+off, p.y+off, p.z-off); 
	glVertex3f(p.x-off, p.y+off, p.z-off); 
	glVertex3f(p.x-off, p.y-off, p.z-off); 
	glVertex3f(p.x+off, p.y-off, p.z-off); 
	//	Top
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(p.x+off, p.y+off, p.z+off); 
	glVertex3f(p.x-off, p.y+off, p.z+off); 
	glVertex3f(p.x+off, p.y+off, p.z-off); 
	glVertex3f(p.x-off, p.y+off, p.z-off); 
	//	Left
	glColor3f(0.0, 1.0, 0.0);
	glVertex3f(p.x-off, p.y+off, p.z+off); 
	glVertex3f(p.x-off, p.y-off, p.z+off); 
	glVertex3f(p.x-off, p.y+off, p.z-off); 
	glVertex3f(p.x-off, p.y-off, p.z-off); 
	//	Bottom
	glColor3f(1.0, 0.0, 0.0);
	glVertex3f(p.x+off, p.y-off, p.z+off); 
	glVertex3f(p.x-off, p.y-off, p.z+off); 
	glVertex3f(p.x+off, p.y-off, p.z-off); 
	glVertex3f(p.x-off, p.y-off, p.z-off); 

	glEnd();
	glFlush();
	glutSwapBuffers();
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


