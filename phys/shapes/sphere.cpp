#include "sphere.h"
#include "math.h"

double Sphere::volume()
{
	return M_PI*radius*radius;
}

double Sphere::moment()
{
	// For a solid sphere: I = 2/5MR^2
	return (0.4)*mass*radius*radius;
}

double Sphere::density()
{
	return mass/volume();
}
