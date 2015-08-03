#include "phys.h"


///////////////////
// Helper Functions 
///////////////////

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



void resetForces(std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++)
	{
		v[i]->t_forces.x = 0;
		v[i]->t_forces.y = 0;
		v[i]->t_forces.z = 0;
		v[i]->r_forces.x = 0;
		v[i]->r_forces.y = 0;
		v[i]->r_forces.z = 0;
	}
}


////////////////////
// Gravity Functions 
////////////////////

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
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		// Affect by all other elements except itself
		for(int j=0; j<v.size(); j++){
			if(i != j)
				gravPull(v[j], v[i]);
		}
	}
}

void gravAllMass(double uniMass, cart uniMassDist, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		// Affect by universal mass
		gravPull(uniMass, uniMassDist, v[i]);
	}
}


//////////////////////
// Collision functions
//////////////////////

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
	#pragma omp parallel for schedule(static)
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
	// We can either assume collision already happened and move the objects
	// to their correct spots, correcting their velocity vectors OR we can 
	// set up an impulse in the correct direction that will have that effect
	// Rule 1 - conserve momentum
	// Rule 2 - conserve KE with respect to dampingConst

	// Get distance b/t shapes
	// Compute I of each shape using distance
	
	// Get tangent vector(s?)
	// Compute angular velocities using tangent vectors
	// Find angular momentums using the angular vel of each shape
		// Remember to add 
	
	
}


///////////////////////
// Simulation Functions 
///////////////////////

void t_advancePos(double t, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		double mass = v[i]->mass;

		cart vel = {v[i]->t_velocity.x,
					v[i]->t_velocity.y,
					v[i]->t_velocity.z};

		cart accel = {	v[i]->t_forces.x/mass,
						v[i]->t_forces.y/mass,
						v[i]->t_forces.z/mass};

		// Using parabolic motion df = d0 + vt + 0.5at^2
		v[i]->t_position.x += vel.x*t + 0.5*accel.x*t*t; 
		v[i]->t_position.y += vel.y*t + 0.5*accel.y*t*t; 
		v[i]->t_position.z += vel.z*t + 0.5*accel.z*t*t; 
	}
}

// Move one timestep using the rotational forces (torques)  on all the objects
void r_advancePos(double t, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		double mass = v[i]->mass;

		cart rVel = {	v[i]->r_velocity.x,
						v[i]->r_velocity.y,
						v[i]->r_velocity.z};

		cart rAccel = {	v[i]->r_forces.x/v[i]->momentCM().x,
                        v[i]->r_forces.y/v[i]->momentCM().y,
                        v[i]->r_forces.z/v[i]->momentCM().z,};

		// Using parabolic motion thetaf = theta0 + omega*t + 0.5*alpha*t^2
		v[i]->r_position.x += rVel.x*t + 0.5*rAccel.x*t*t; 
		v[i]->r_position.y += rVel.y*t + 0.5*rAccel.y*t*t; 
		v[i]->r_position.z += rVel.z*t + 0.5*rAccel.z*t*t; 
	}
}

void advancePosAndReset(double t, std::vector<Shape*> v)
{
	t_advancePos(t, v);
	r_advancePos(t, v);

	// Reset force vectors
	resetForces(v);
}

void wrapWorld(cart worldLimits, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++)
	{
		// Positive world limit breaches
		if(v[i]->t_position.x > worldLimits.x){
			// Original math
			//v[i]->t_pos.x = -worldLimits.x + (v[i]->t_pos.x - worldLimits.x);
			// Simplified math
			v[i]->t_position.x = -2*worldLimits.x + v[i]->t_position.x;
		}
		// Negative world limit breaches
		else if(v[i]->t_position.x < -worldLimits.x){
			v[i]->t_position.x = 2*worldLimits.x + v[i]->t_position.x;
		}

		if(v[i]->t_position.y > worldLimits.y){
			v[i]->t_position.y = -2*worldLimits.y + v[i]->t_position.y;
		}
		else if(v[i]->t_position.y < -worldLimits.y){
			v[i]->t_position.y = 2*worldLimits.y + v[i]->t_position.y;
		}

		if(v[i]->t_position.z > worldLimits.z){
			v[i]->t_position.z = -2*worldLimits.z + v[i]->t_position.z;
		}
		else if(v[i]->t_position.z < -worldLimits.z){	
			v[i]->t_position.z = 2*worldLimits.z + v[i]->t_position.z;
		}
	}

}

void advanceSim(double t, std::vector<Shape*> v)
{
	// MAKE SURE THIS IS THE LEAST ERROR-PRONE ORDER

	// Physicall influences (gravity/magnetism etc)
	//gravAllShapes(v);

	// Update position of all shapes
	advancePosAndReset(t, v);

	// Detect and resolve all collisions
	//collideAndResolve(v);

	// If worldwrap is on, worldwrap all objects
	cart lims = {100, 100, 100};
	wrapWorld(lims, v);
}
