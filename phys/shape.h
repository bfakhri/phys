#ifndef SHAPE_H
#define SHAPE_H

#include "drawingParams.h"	// For drawing params
#include "mather.h"			// For cart type	

#include <GL/glut.h>	// For draw() function
#include <GL/gl.h>		// For draw() function
#include <stdlib.h>		// For rand()

// scalar quantities are preceded by nothing
// vector quantities are preceded by either a "t_" for translational or
// a "r_" for rotaitional

// This should be as concise as possible so we will use the cartesian structure as much as possible

// All units should adhere to SI standards

// Shape class (used to be the 'mass' class)
class Shape
{
	public:
		// Scalar quantities
		double mass;	// kilograms

		// Vector quantities
		// 	Translational
		cart t_position;	// meters from origin
		cart t_velocity;	// m/s
		cart t_forces;		// newtons

		//	Rotational
		cart r_position;	// radians from origin
		cart r_velocity;	// rad/sec
		cart r_forces;		// torques - Nm

	
		// Constructors
		Shape();
		Shape(double mass, cart tPos, cart tVel, cart rPos, cart rVel);	// Forces are missing because we will init to zero most probably
		
		// Mutators
		void t_addForce(cart force);	// Adds a translational force
		void r_addForce(cart force);	// Addes a rotational force
		void resetForces();				// Resets ALL forces to zero
		
		// Drawing stuff
		void draw();				// Sets up the drawing scheme by moving to the right 
									//	place and rotating. Then calls drawShape()
		
		// Abstract methods that other shapes MUST define
		virtual void drawShape()=0;	// Does the part of draw that is specific to the shape
		virtual double volume()=0;	// Returns volume of sphere
		virtual cart momentCM()=0;	// Returns moment of inertia of shape through center of mass
		cart moment(cart d);		// Returns moment of inertia of shape through a parallel 
						//	axis a distance 'd' from the axis at center of mass
						// 	Not virtual because it will be the same for all subclasses
		virtual double density()=0;	// Returns density of sphere
		virtual double boundingSphere()=0;// Returns the radius of the bounding sphere of an object
		virtual double boundingBox()=0;	// Returns the length of the bounding cube of an object
						// This may need to be more elaborate (give more than a side length)

};
				
#endif
