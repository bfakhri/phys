#include "sphere.h"
#include "math.h"


///////////////
// Constructors
///////////////

Sphere::Sphere()
{
	radius = 1;
}

Sphere::Sphere(double r, double sMass, cart tPos, cart tVel, cart rPos, cart rVel): Shape(sMass, tPos, tVel, rPos, rVel)
{
	radius = r; 
}


/////////////////
// Drawing Functs
/////////////////

void Sphere::drawShape()
{
	// Mem-leak!?!?!?
	GLUquadric* quad = gluNewQuadric();					// make a quadric
	gluQuadricDrawStyle(quad, GLU_POINT);				// This changes the drawing style
	gluSphere(quad, radius, DEF_SLICES, DEF_STACKS);	// Draws the sphere
}


/////////////////
// Physics Functs
/////////////////

double Sphere::volume()
{
	return M_PI*radius*radius;
}

cart Sphere::momentCM()
{
	// For a solid sphere: I = 2/5MR^2
	cart mmnt = {	(0.4)*mass*radius*radius,
			(0.4)*mass*radius*radius,
			(0.4)*mass*radius*radius};
	return mmnt;
}

double Sphere::density()
{
	return mass/volume();
}

double Sphere::boundingSphere()
{
	return radius;
}

// For a sphere this is the same as boundingSphere
// For a cube this would not be the same
double Sphere::boundingBox()
{
	return 2*radius;
}
