#ifndef SHAPE_H
#define SHAPE_H
#include "all_includes.h"

// scalar quantities are preceded by nothing
// vector quantities are preceded by either a "t_" for translational or
// a "r_" for rotaitional

// This should be as concise as possible so we will use the cartesian structure as much as possible

// All units should adhere to SI standards

Class Shape
{
	private:
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

	public:
		// Constructors
		Shape();
		Shape(double mass, cart tPos, cart tVel, cart rPos, cart rVel);	// Forces are missing because we will init to zero most probably
		
		// Mutators
		void t_addForce(cart force);	// Adds a translational force
		void r_addForce(cart force);	// Addes a rotational force
		void resetForces();		// Resets ALL forces to zero
		
		void t_updatePos(double t);	// Updates translational position
		void r_updatePos(double t);	// Updates rotational position

		void updatePosResetForces(double t);	// Updates both trans and rot positions and resets forces
}
		
		

#endif
