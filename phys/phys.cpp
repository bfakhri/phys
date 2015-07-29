#include "phys.h"


double distance(Shape s1, Shape s2)
{	
	cart s2tos1 = {s1->t_pos.x - s2->t_pos.x, 
			s1->t_pos.y - s2->t_pos.y, 
			s1->t_pos.z - s2->t_pos.z};
	return sqrt(s;
}

double gravForce(double m1, double m2)
{
	return ;
}

void gravInfluence(Shape m1, Shape m2)
{
	// Get direction vector
	cart m2tom1 = {m1->t_pos.x - m2->t_pos.x, 
			m1->t_pos.y - m2->t_pos.y, 
			m1->t_pos.z - m2->t_pos.z};

	// Get magnitude of the gravitational force
	double forceMag = gravForce(m1.mass(), m2.mass());

	// Influence m2 by that force 
	m2tom1.x *= forceMag;
	m2tom1.y *= forceMag;
	m2tom1.z *= forceMag;
	m2->addForce(m2tom1);
	
}

void gravity(double uniMass, cart uniMassDist, std::vector<Shape*> v)
{
	// Check if universal mass exists
	// Affect all masses by this one if exists
	if(uniMass > 0)
	{
		for(int i=0; i<v.size(); i++){
			// Affect by universal mass
			// Affect by all other elements except itself
		}
	}else{
		for(int i=0; i<v.size(); i++){
			// Affect by universal mass
			// Affect by all other elements except itself
		}
	}
}

bool collide(Shape s1, Shape s2)
{

}

void collideAndResolve(std::vector<Shape*> v)
{

}

void resolveCollision(Shape s1, Shape s2, double dampingConst)
{

}

void updatePosWrap(cart worldLimits, std::vector<Shape*> v)
{

}

void advanceSim(double t, std::vector<Shape*> v)
{
	// Update position of all shapes

	// Detect and resolve all collisions

	// If worldwrap is on, worldwrap all objects
}
