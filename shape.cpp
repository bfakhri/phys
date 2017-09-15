#include "shape.h"



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
	t_pInf.x = 0;
	t_pInf.y = 0;
	t_pInf.z = 0;

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
	r_pInf.x = 0;
	r_pInf.y = 0;
	r_pInf.z = 0;

	// Visual
	collides = false;
	color.x = 0.5;
	color.y = 0.5;
	color.z = 0.5;
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
	t_pInf.x = 0;
	t_pInf.y = 0;
	t_pInf.z = 0;
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
	r_pInf.x = 0;
	r_pInf.y = 0;
	r_pInf.z = 0;

	// Visual
	collides = false;
	color.x = 0.5;
	color.y = 0.5;
	color.z = 0.5;
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

void Shape::t_addMomentum(cart p)
{
	t_pInf.x += p.x;
	t_pInf.y += p.y;
	t_pInf.z += p.z;
}		

void Shape::r_addMomentum(cart p)
{
	r_pInf.x += p.x;
	r_pInf.y += p.y;
	r_pInf.z += p.z;
}		

void Shape::r_addMomentum(cart p);		

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


// Resets all forces to zero
void Shape::resetMomentum()
{
	t_pInf.x = 0;
	t_pInf.y = 0;
	t_pInf.z = 0;
	r_pInf.x = 0;
	r_pInf.y = 0;
	r_pInf.z = 0;

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

	// Set color to color of shape
	if(collides)
		glColor3f(0.0, 0, 0);
	else
		glColor3f(color.x, color.y, color.z);
	
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


////////////////
// Helper Functs
////////////////


void populateShapeVector(std::vector<Shape*> v)
{
	// write this
}


void drawBoundaries(cart origin, cart min, cart max)
{
	glPushMatrix(); 
	// Go to physics origin
	glTranslatef(origin.x, origin.y, origin.z); 
	
	glBegin(GL_QUADS);
	
	//	Back	
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, max.y, min.z); 
	glVertex3f(min.x, max.y, min.z); 
	glVertex3f(min.x, min.y, min.z); 
	glVertex3f(max.x, min.y, min.z); 
	//	Right
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(max.x, max.y, max.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, max.y, min.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, min.y, min.z); 
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(max.x, min.y, max.z); 
	//	Top
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(max.x, max.y, max.z); 
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(min.x, max.y, max.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, max.y, min.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, max.y, min.z); 
	//	Left
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, max.y, min.z); 
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(min.x, max.y, max.z); 
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(min.x, min.y, max.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, min.y, min.z); 
	/*//	Bottom
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, min.y, min.z); 
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(max.x, min.y, max.z); 
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(min.x, min.y, max.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, min.y, min.z); 
	*/
	//	Bottom
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, min.y, min.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, min.y, min.z); 
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(min.x, min.y, max.z); 
	glColor3f(0.1, 0.1, 0.8);
	glVertex3f(max.x, min.y, max.z); 

	glEnd();
	glPopMatrix();
}
