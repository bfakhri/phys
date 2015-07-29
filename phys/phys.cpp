#include "phys.h"


double distance(cart c1, cart c2)
{	
	cart c2toc1 = {	c1.x - c2.x, 
			c1.y - c2.y, 
			c1.z - c2.z};
	return sqrt(c2toc1.x*c2toc1.x + c2toc1.y*c2toc1.y + c2toc1.z*c2toc1.z);
}

double distance(Shape* s1, Shape* s2)
{	
	cart c2toc1 = {s1->t_position.x - s2->t_position.x, 
			s1->t_position.y - s2->t_position.y, 
			s1->t_position.z - s2->t_position.z};
	return sqrt(c2toc1.x*c2toc1.x + c2toc1.y*c2toc1.y + c2toc1.z*c2toc1.z);
}

double gravForce(double m1, double m2, double dist)
{
	return G_CONST*m1*m2/(dist*dist);
}

void gravInfluenceShape(Shape* m1, Shape* m2)
{
	// Get direction vector
	cart m2tom1 = {m1->t_position.x - m2->t_position.x, 
			m1->t_position.y - m2->t_position.y, 
			m1->t_position.z - m2->t_position.z};

	// Get magnitude of the gravitational force
	double forceMag = gravForce(m1->mass, m2->mass, distance(m1, m2));

	// Influence m2 by that force 
	m2tom1.x *= forceMag;
	m2tom1.y *= forceMag;
	m2tom1.z *= forceMag;
	m2->t_addForce(m2tom1);
	
}

void gravInfluenceMass(double uniMass, cart uniMassDist, Shape* s)
{
	// Get direction vector
	cart shapeToMass = {	uniMassDist.x - s->t_position.x, 
				uniMassDist.y - s->t_position.y, 
				uniMassDist.z - s->t_position.z};

	// Get magnitude of the gravitational force
	double forceMag = gravForce(uniMass, s->mass, distance(s->t_position, uniMassDist));

	// Influence m2 by that force 
	shapeToMass.x *= forceMag;
	shapeToMass.y *= forceMag;
	shapeToMass.z *= forceMag;
	s->t_addForce(shapeToMass);
}

void gravity(double uniMass, cart uniMassDist, std::vector<Shape*> v)
{
	// Check if universal mass exists
	// Affect all masses by this one if exists
	if(uniMass > 0)
	{
		for(int i=0; i<v.size(); i++){
			// Affect by universal mass
			gravInfluenceMass(uniMass, uniMassDist, v[i]);
			// Affect by all other elements except itself
			for(int j=0; j<v.size(); j++){
				gravInfluenceShape(v[j], v[i]);
			}
		}
	}else{
		for(int i=0; i<v.size(); i++){
			// Affect by all other elements except itself
			for(int j=0; j<v.size(); j++){
				gravInfluenceShape(v[j], v[i]);
			}
		}
	}
}

bool collide(Shape* s1, Shape* s2)
{
	// Need to figure out for all shape combos
}

bool collide(Sphere* s1, Sphere* s2)
{
	if(distance(s1, s2) < (s1->radius + s2->radius))
		return true; 
	else
		return false; 
}

void collideAndResolve(std::vector<Shape*> v)
{

}

void collideAndResolve(std::vector<Sphere*> v)
{
	// Make sure this cycles through ALL pairs
	for(int i=0; i<v.size(); i++){
		for(int j=i; j<v.size(); j++){
			if(collide(v[i], v[j])){
				resolveCollision(v[i], v[j], 0);
			}
		}
	}
}
void resolveCollision(Shape* s1, Shape* s2, double dampingConst)
{
	// Rule 1 - conserve momentum
	// Rule 2 - conserve KE with respect to dampingConst
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
