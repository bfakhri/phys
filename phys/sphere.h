// This file describes the Sphere subclass of Shape
// It has the required functions volume() and moment of inertia()
// as well as properties relevant to a sphere

#ifndef SPHERE_H
#define SPHERE_H

#include "shape.h"

// Glut files
#include <GL/glut.h>
#include <GL/gl.h>

#define SPHERE_COLOR 1.0, 1.0, 1.0

class Sphere : public Shape 
{
	public:
		double radius;	
	
		///////////////
		// Constructors
		///////////////

		Sphere();
		Sphere(double r, double sMass, cart tPos, cart tVel, cart rPos, cart rVel);

		/////////////////
		// Drawing Functs
		/////////////////

		void drawShape();
		
		/////////////////
		// Physics Functs
		/////////////////

		double volume();		// Returns volume of sphere
		cart momentCM();		// Returns moment of inertia of sphere
		double density();		// Returns density of sphere
		double boundingSphere();// Returns the radius of the bounding sphere of an object
		double boundingBox();	// Returns the length of the bounding cube of an object

};

////////////////
// Helper Functs
////////////////

// Make random shape with param constraints
Shape* randomShape();

// Make random shape with param constraints
Shape* randomShape(double radMin, double radMax, double massMin, double massMax, cart tMaxPos, cart tMaxVel);
#endif
