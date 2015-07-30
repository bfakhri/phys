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

Shape* randomShape(double radMin, double radMax, double minMass, double maxMass, cart tMaxPos,cart tMaxVel, cart rMaxPos, cart rMaxVel)
{
	// This function might need to scale the values to make sure 
	// we don't get false ceilings for when we need values larger
	// than RAND_MAX
	double radius = (rand()%((unsigned int)(radMax-radMin)))+radMin;
	double mass = (rand()%((unsigned int)(maxMass-minMass)))+minMass;
	cart tPos = {rand()%((unsigned int)tMaxPos.x),
			rand()%((unsigned int)tMaxPos.y),
			rand()%((unsigned int)tMaxPos.z)};
	cart tVel = {rand()%((unsigned int)tMaxVel.x),
			rand()%((unsigned int)tMaxVel.y),
			rand()%((unsigned int)tMaxVel.z)};
	cart rPos = {rand()%((unsigned int)rMaxPos.x),
			rand()%((unsigned int)rMaxPos.y),
			rand()%((unsigned int)rMaxPos.z)};
	cart rVel = {rand()%((unsigned int)rMaxVel.x),
			rand()%((unsigned int)rMaxVel.y),
			rand()%((unsigned int)rMaxVel.z)};

	// This must be generalized so that any shape type is
	// possible. Not just Sphere
	return new Sphere(radius, mass, tPos, tVel, tPos, rVel); 
}

