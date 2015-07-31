#include "commonFunctions.h"

void populateShapeVector()
{
	// write this
}

Shape* randomShape()
{
	// This function might need to scale the values to make sure 
	// we don't get false ceilings for when we need values larger
	// than RAND_MAX
	double radius = (rand()%10);
	double mass = (rand());
	cart tPos = {	rand()%100,
			rand()%100,
			rand()%100};
	cart tVel = {	rand()%10,
			rand()%10,
			rand()%10};
	cart rPos = {	rand()%100,
			rand()%100,
			rand()%100};
	cart rVel = {	rand()%10,
			rand()%10,
			rand()%10};

	// This must be generalized so that any shape type is
	// possible. Not just Sphere
	return new Sphere(radius, mass, tPos, tVel, tPos, rVel); 
}

Shape* randomShape(double radMin, double radMax, double massMin, double massMax, cart tMaxPos,cart tMaxVel)
{
	cart zeroes = {0, 0, 0}; 	

	double radius = (rand()*(radMax+radMin)/RAND_MAX + radMin);
	double mass = (rand()*(massMax+massMin)/RAND_MAX + massMin);

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


