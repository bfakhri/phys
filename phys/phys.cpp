#include "phys.h"


double distance(Shape* s1, Shape* s2)
{	
	cart s2tos1 = {s1->t_position.x - s2->t_position.x, 
			s1->t_position.y - s2->t_position.y, 
			s1->t_position.z - s2->t_position.z};
	return sqrt(s2tos1.x*s2tos1.x + s2tos1.y*s2tos1.y + s2tos1.z*s2tos1.z);
}

double gravForce(double m1, double m2, double dist)
{
	return G_CONST*m1*m2/(dist*dist);
}

void gravInfluence(Shape* m1, Shape* m2)
{
	// Get direction vector
	cart m2tom1 = {m1->t_position.x - m2->t_position.x, 
			m1->t_position.y - m2->t_position.y, 
			m1->t_position.z - m2->t_position.z};

	// Get magnitude of the gravitational force
	double forceMag = gravForce(m1.mass(), m2.mass(), distance(m1, m2));

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
			//v[i]
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
