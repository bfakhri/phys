// This file describes the Sphere subclass of Shape
// It has the required functions volume() and moment of inertia()
// as well as properties relevant to a sphere

#ifndef SPHERE_H
#define SPHERE_H

#include "shape.h"

class Sphere : public Shape		// Is public the right modifier here? 
{
	private:
		double radius;		// The radius of the sphere
	
	public:
		// Need a constructor
		// Figure out how to do this elegantly
		Sphere();	// Constructor

		double volume();	// Returns volume of sphere
		double moment();	// Returns moment of inertia of sphere
		double density();	// Returns density of sphere

};

#endif
