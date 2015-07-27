#ifndef SHAPE_H
#define SHAPE_H
#include "all_includes.h"

// scalar quantities are preceded by nothing
// vector quantities are preceded by either a "t_" for translational or
// a "r_" for rotaitional

// This should be as concise as possible so we will use the cartesian structure as much as possible

Class Shape
{
	private:
		// Scalar quantities
		double mass;

		// Vector quantities
		// 	Translational
		cart t_position;
		cart t_velocity;
		cart t_forces;

		//	Rotational
		cart r_position;
		cart r_velocity;
		cart r_forces;

	public:
		// Constructors
		Shape();
		Shape(double mass, cart tPos, cart tVel, cart rPos, cart rVel);	// Forces are missing because we will init to zero most probably
		
		// Mutators
		void t_addForce(cart force);	// Adds a translational force
		void r_addForce(cart force);	// Addes a rotational force
		void resetForces();		// Resets ALL forces to zero
		
		void t_updatePos();		// Updates translational position
		void r_updatePos();		// Updates rotational position

		void updatePosResetForces();	// Updates both trans and rot positions and resets forces
}
		

		
		

#endif
