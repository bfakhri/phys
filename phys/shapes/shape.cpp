#include "shape.h"

#include <vector>	// For vector to hold all shapes

std::vector<Shape> shapeVector;

void populateShapeVector()
{
	// write this
}

void drawAllShapes()
{
	for(int i=0; i<shapeVector.size(); i++){
		shapeVector[i].draw();
	}
}

Shape::Shape()
{
	// Scalar quantities
	mass = 1; 

	// Vector quantities
	// 	Translational
	t_position.x = 0;
	t_position.y = 0;
	t_position.z = 0;
	t_velocity.x = 0;
	t_velocity.y = 0;
        t_velocity.z = 0;
	t_forces.x = 0;
	t_forces.y = 0;
        t_forces.z = 0;
	//	Rotational
	r_position.x = 0;
	r_position.y = 0;
	r_position.z = 0;
	r_velocity.x = 0;
	r_velocity.y = 0;
        r_velocity.z = 0;
	r_forces.x = 0;
	r_forces.y = 0;
        r_forces.z = 0;
};

Shape::Shape(double sMass, cart tPos, cart tVel, cart rPos, cart rVel)
{
	// Scalar quantities
	mass = sMass; 

	// Vector quantities
	// 	Translational
	t_position.x = tPos.x;
	t_position.y = tPos.y;
	t_position.z = tPos.z;
	t_velocity.x = tVel.x;
	t_velocity.y = tVel.y;
        t_velocity.z = tVel.z;
	t_forces.x = 0;
	t_forces.y = 0;
        t_forces.z = 0;
	//	Rotational
	r_position.x = rPos.x;
	r_position.y = rPos.y;
	r_position.z = rPos.z;
	r_velocity.x = rVel.x;
	r_velocity.y = rVel.y;
        r_velocity.z = rVel.z;
	r_forces.x = 0;
	r_forces.y = 0;
        r_forces.z = 0;
};

void Shape::t_addForce(cart force)
{
	t_forces.x += force.x;
	t_forces.y += force.y;
        t_forces.z += force.z;
};


void Shape::r_addForce(cart force)
{
	r_forces.x += force.x;
	r_forces.y += force.y;
        r_forces.z += force.z;
};

// Resets all forces to zero
void Shape::resetForces()
{
	t_forces.x = 0;
	t_forces.y = 0;
        t_forces.z = 0;
	r_forces.x = 0;
	r_forces.y = 0;
        r_forces.z = 0;

};

// Updates the shape by moving forward in time by t seconds
void Shape::t_updatePos(double t)
{
	t_position.x += t_velocity.x*t;
	t_position.y += t_velocity.y*t;
	t_position.z += t_velocity.z*t;
};

// Updates the shape by moving forward in time by t seconds
void Shape::r_updatePos(double t)
{
	r_position.x += r_velocity.x*t;
	r_position.y += r_velocity.y*t;
	r_position.z += r_velocity.z*t;
};

void Shape::updatePosResetForces(double t)
{
	t_updatePos(t);
	r_updatePos(t);
	resetForces();
};

void Shape::draw()
{
	// Call drawing function from draw.cpp with elements from this shape
}

