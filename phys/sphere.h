// This file describes the Sphere subclass of Shape
// It has the required functions volume() and moment of inertia()
// as well as properties relevant to a sphere

#ifndef SPHERE_H
#define SPHERE_H

#include "shape.h"

class Sphere : public Shape		// Is public the right modifier here? 
{
	public:
		double radius;		// The radius of the sphere
	
		// Default
		Sphere();
		// With params
		Sphere(double r, double sMass, cart tPos, cart tVel, cart rPos, cart rVel);

		
		// Physics related methods
		double volume();	// Returns volume of sphere
		double moment();	// Returns moment of inertia of sphere
		double density();	// Returns density of sphere
		double boundingSphere();// Returns the radius of the bounding sphere of an object
		double boundingBox();	// Returns the length of the bounding cube of an object

};

#endif
