#include "shape.h"


////////////////
// Helper Functs
////////////////

void populateShapeVector(std::vector<Shape*> v)
{
	// write this
}

Shape* randomShape()
{
	// This function might need to scale the values to make sure 
	// we don't get false ceilings for when we need values larger
	// than RAND_MAX
	double radius = ((double)(rand()%10));
	double mass = ((double)rand());
	cart tPos = {	(double)(rand()%100),
					(double)(rand()%100),
					(double)(rand()%100)};
	cart tVel = {	(double)(rand()%10),
					(double)(rand()%10),
					(double)(rand()%10)};
	cart rPos = {	(double)(rand()%100),
					(double)(rand()%100),
					(double)(rand()%100)};
	cart rVel = {	(double)(rand()%10),
					(double)(rand()%10),
					(double)(rand()%10)};

	// This must be generalized so that any shape type is
	// possible. Not just Sphere
	return new Sphere(radius, mass, tPos, tVel, tPos, rVel); 
}

Shape* randomShape(double radMin, double radMax, double massMin, double massMax, cart tMaxPos,cart tMaxVel)
{
	cart zeroes = {0, 0, 0}; 	

	double radius = (rand()*(radMax+radMin)/RAND_MAX + radMin);
	double mass = (rand()*(massMax+massMin)/RAND_MAX + massMin);

	cart tPos = {	(rand()*(2*tMaxPos.x)/RAND_MAX - tMaxPos.x), 
					(rand()*(2*tMaxPos.y)/RAND_MAX - tMaxPos.y), 
					(rand()*(2*tMaxPos.z)/RAND_MAX - tMaxPos.z)};

	cart tVel = {	(rand()*(2*tMaxVel.x)/RAND_MAX - tMaxVel.x), 
					(rand()*(2*tMaxVel.y)/RAND_MAX - tMaxVel.y), 
					(rand()*(2*tMaxVel.z)/RAND_MAX - tMaxVel.z)};


	// This must be generalized so that any shape type is
	// possible. Not just Sphere
	return new Sphere(radius, mass, tPos, tVel, zeroes, zeroes);  
}


///////////////
// Constructors
///////////////

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



///////////
// Mutators 
///////////

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



/////////////////
// Drawing Functs
/////////////////

void Shape::draw(cart origin)
{
	// Setup the draw
	glPushMatrix(); 
	// Push everything forward so origin in not in the camera	
	glTranslatef(origin.x, origin.y, origin.z); 
	glTranslatef(t_position.x, t_position.y, t_position.z); 
	glRotatef(r_position.x, 1, 0 , 0); 
	glRotatef(r_position.y, 0, 1 , 0); 
	glRotatef(r_position.z, 0, 0 , 1); 

	// Actually draw it
	drawShape();

	// Reset the matrix
	glPopMatrix(); 
}


/////////////////
// Physics Functs
/////////////////

cart Shape::moment(cart d)
{
	cart mmnt = {	momentCM().x + mass*(d.x*d.x),
			momentCM().y + mass*(d.y*d.y),
			momentCM().z + mass*(d.z*d.z)};

	return mmnt;
}

