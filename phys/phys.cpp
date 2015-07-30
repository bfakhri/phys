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
	cart c1 = s1->t_position;
	cart c2 = s2->t_position;
	return distance(c1, c2);  
}

double gravForce(double m1, double m2, double dist)
{
	return G_CONST*m1*m2/(dist*dist);
}

void gravPull(Shape* m1, Shape* m2)
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

void gravPull(double uniMass, cart uniMassDist, Shape* s)
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

void gravAllShapes(std::vector<Shape*> v)
{
	for(int i=0; i<v.size(); i++){
		// Affect by all other elements except itself
		for(int j=0; j<v.size(); j++){
			if(i != j)
				gravInfluenceShape(v[j], v[i]);
		}
	}
}

void gravAllMass(double uniMass, cart uniMassDist, std::vector<Shape*> v);
{
	for(int i=0; i<v.size(); i++){
		// Affect by universal mass
		gravPull(uniMass, uniMassDist, v[i]);
	}
}

// We may want to make this function more general by adding a criteria
// parameter to determine how we decide a collision has occured (sphere/bounding box/etc)
bool collide(Shape* s1, Shape* s2)
{
	// All shapes are treated as spheres for collisions as of now
	if(distance(s1, s2) < (s1->boundingSphere() + s2->boundingSphere()))
		return true; 
	else
		return false; 
}


void collideAndResolve(std::vector<Shape*> v)
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

// May need to delete this to keep things somewhat general
/*
void collideAndResolve(std::vector<Sphere*> v)
{
}
*/

void resolveCollision(Shape* s1, Shape* s2, double dampingConst)
{
	// Rule 1 - conserve momentum
	// Rule 2 - conserve KE with respect to dampingConst
}

void t_advancePos(double t, std::vector<Shape*> v)
{

}

// Move one timestep using the rotational forces (torques)  on all the objects
void r_advancePos(double t, std::vector<Shape*> v)
{

}

// Move one timestep both translational and rotational positions 
void advancePos(double t, std::vector<Shape*> v)
{
	t_advancePos(t, v);
	r_advancePos(t, v);
}

void wrapWorld(cart worldLimits, std::vector<Shape*> v)
{
	for(int i=0; i<v.size(); i++)
	{
		// Positive world limit breaches
		if(v[i]->t_pos.x > worldLimits.x){
			// Original math
			//v[i]->t_pos.x = -worldLimits.x + (v[i]->t_pos.x - worldLimits.x);
			// Simplified math
			v[i]->t_pos.x = -2*worldLimits.x + v[i]->t_pos.x;
		}
		// Negative world limit breaches
		else if(v[i]->t_pos.x < -worldLimits.x)	
			v[i]->t_pos.x = 2*worldLimits.x + v[i]->t_pos.x;
		}

		if(v[i]->t_pos.y > worldLimits.y){
			v[i]->t_pos.y = -2*worldLimits.y + v[i]->t_pos.y;
		}
		else if(v[i]->t_pos.y < -worldLimits.y)	
			v[i]->t_pos.y = 2*worldLimits.y + v[i]->t_pos.y;
		}

		if(v[i]->t_pos.z > worldLimits.z){
			v[i]->t_pos.z = -2*worldLimits.z + v[i]->t_pos.z;
		}
		else if(v[i]->t_pos.z < -worldLimits.z)	
			v[i]->t_pos.z = 2*worldLimits.z + v[i]->t_pos.z;
		}


}

void advanceSim(double t, std::vector<Shape*> v)
{
	// Update position of all shapes
	cart blankCart;
	gravity(0, blankCart, v);
	// Detect and resolve all collisions

	// If worldwrap is on, worldwrap all objects
}
