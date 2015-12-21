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
	gluQuadricDrawStyle(quad, GLU_FILL);				// This changes the drawing style
	//float depthColor = (t_position.z+10)/20;
	//glColor3f(1.0, 0, 0); //depthColor, depthColor, depthColor); 
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


////////////////
// Helper Functs
////////////////

Shape* randomShape()
{
	// This function might need to scale the values to make sure 
	// we don't get false ceilings for when we need values larger
	// than RAND_MAX
	double radius = ((double)(rand()%10));
	double mass = ((double)rand());
	cart tPos = {	(double)(rand()%100),
					(double)(rand()%100),
					(double)(rand()%100)};
	cart tVel = {	(double)(rand()%10),
					(double)(rand()%10),
					(double)(rand()%10)};
	cart rPos = {	(double)(rand()%100),
					(double)(rand()%100),
					(double)(rand()%100)};
	cart rVel = {	(double)(rand()%10),
					(double)(rand()%10),
					(double)(rand()%10)};

	// This must be generalized so that any shape type is
	// possible. Not just Sphere
	return new Sphere(radius, mass, tPos, tVel, tPos, rVel); 
}

Shape* randomShape(double radMin, double radMax, double massMin, double massMax, cart tMaxPos,cart tMaxVel)
{
	cart zeroes = {0, 0, 0}; 	

	double radius; 
	radius = (rand()*(radMax-radMin)/RAND_MAX + radMin);

	double mass;
	mass = (rand()*(massMax-massMin)/RAND_MAX + massMin);

	cart tPos = {	(rand()*(2*tMaxPos.x)/RAND_MAX - tMaxPos.x), 
					(rand()*(2*tMaxPos.y)/RAND_MAX - tMaxPos.y), 
					(rand()*(2*tMaxPos.z)/RAND_MAX - tMaxPos.z)};

	cart tVel = {	(rand()*(2*tMaxVel.x)/RAND_MAX - tMaxVel.x), 
					(rand()*(2*tMaxVel.y)/RAND_MAX - tMaxVel.y), 
					(rand()*(2*tMaxVel.z)/RAND_MAX - tMaxVel.z)};


	// This must be generalized so that any shape type is
	// possible. Not just Sphere
	return new Sphere(radius, mass, tPos, tVel, zeroes, zeroes);  
}
