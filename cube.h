// This file describes the square subclass of Shape

#ifndef CUBE_H
#define CUBE_H

#include "shape.h"

class Cube : public Shape		// Is public the right modifier here? 
{
	public:
		double sideLength;	// The radius of the Cube
	
		///////////////
		// Constructors
		///////////////

		Cube();
		Cube(double r, double sMass, cart tPos, cart tVel, cart rPos, cart rVel);

		/////////////////
		// Drawing Functs
		/////////////////

		void drawShape();
		
		/////////////////
		// Physics Functs
		/////////////////
		
		double volume();		// Returns volume of Cube
		cart momentCM();		// Returns moment of inertia of Cube
		double density();		// Returns density of Cube
		double boundingSphere();// Returns the radius of the bounding sphere of an object
		double boundingBox();	// Returns the length of the bounding cube of an object

};

#endif
