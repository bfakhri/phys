#include "sphere.h"
#include "math.h"

// Default
Sphere::Sphere()
{
	radius = 1;
}

// Calls parent non-default constructor and passes args to it
Sphere::Sphere(double r, double sMass, cart tPos, cart tVel, cart rPos, cart rVel): Shape(sMass, tPos, tVel, rPos, rVel)
{
	radius = r; 
}

double Sphere::volume()
{
	return M_PI*radius*radius;
}

double Sphere::moment()
{
	// For a solid sphere: I = 2/5MR^2
	cart mmnt = {	(0.4)*mass*radius*radius,
			(0.4)*mass*radius*radius,
			(0.4)*mass*radius*radius};
	return mmnt
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
