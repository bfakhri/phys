#ifndef SHAPE_H
#define SHAPE_H

//#include "engine.h"		// For drawing params
#include "mather.h"		// For cart type	

#include <GL/glut.h>	// For draw() function
#include <GL/gl.h>		// For draw() function
#include <stdlib.h>		// For rand()
#include <vector>

// scalar quantities are preceded by nothing
// vector quantities are preceded by either a "t_" for translational or
// a "r_" for rotaitional

// This should be as concise as possible so we will use the cartesian structure as much as possible

// All units should adhere to SI standards


// Drawing parameters 
// For drawing 3D objects
#define DEF_SLICES 50
#define DEF_STACKS 50


///////////////////
// Class Definition 
///////////////////

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
		cart t_pInf;		// kg*m/s

		//	Rotational
		cart r_position;	// radians from origin
		cart r_velocity;	// rad/sec
		cart r_forces;		// torques - Nm
		cart r_pInf;		// kg*(m^2)*rad/sec

		// Visual
		cart color;			// RGB color of shape
		bool collides;		// Flag for a collision involving this 
							// shape in an iteration
			
		///////////////
		// Constructors
		///////////////
		
		Shape();
		Shape(double mass, cart tPos, cart tVel, cart rPos, cart rVel);	
		
		///////////
		// Mutators 
		///////////

		void t_addForce(cart force);	// Adds a translational force
		void r_addForce(cart force);	// Addes a rotational force
		void t_addMomentum(cart p);		// Adds a translational force
		void r_addMomentum(cart p);		// Addes a rotational momentum
		void resetForces();				// Resets ALL forces to zero
		void resetMomentum();			// Resets ALL momentum to zero
		
		/////////////////
		// Drawing Functs
		/////////////////
		void draw(cart origin);		// Sets up the drawing scheme by moving to the right 
									//	place and rotating. Then calls drawShape()
		
		/////////////////
		// Physics Functs
		/////////////////
		
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
				
////////////////
// Helper Functs
////////////////

// Maybe makes this flexible - can take in shapes from files or random shapes
// depending on the inputs to the function
void populateShapeVector(std::vector<Shape*> v);


// Draws the walls of the simulation
void drawBoundaries(cart origin, cart min, cart max);

#endif
