#include "phys.h"

///////////////////
// Helper Functions 
///////////////////

cart negate(cart c)
{
	cart n = {	-c.x, -c.y, -c.z};
	return n; 
}

double length(cart c)
{
	return sqrt(dotProd(c, c));
}

cart normalize(cart c)
{
	double l = length(c);
	cart normed = c/l;
	return normed;
}

double dotProd(cart c1, cart c2)
{
	return (c1.x*c2.x + c1.y*c2.y + c1.z*c2.z);
}

cart multComponents(cart c1, cart c2)
{
	cart prod = {c1.x*c2.x, c1.y*c2.y, c1.z*c2.z};
	return prod;
}

cart divComponents(cart dividend, cart divisor)
{
	cart quotient = {	dividend.x/divisor.x,
						dividend.y/divisor.y,
						dividend.z/divisor.z};
	return quotient;
}

double distance(cart c1, cart c2)
{	
	cart c2toc1 = c1 - c2;
	return sqrt(dotProd(c2toc1, c2toc1));
}

double distance(Shape* s1, Shape* s2)
{	
	cart c1 = s1->t_position;
	cart c2 = s2->t_position;
	return distance(c1, c2);  
}

void resetForcesAndMomentums(std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		v[i]->t_forces.x = 0;
		v[i]->t_forces.y = 0;
		v[i]->t_forces.z = 0;
		v[i]->r_forces.x = 0;
		v[i]->r_forces.y = 0;
		v[i]->r_forces.z = 0;
		v[i]->t_pInf.x = 0;
		v[i]->t_pInf.y = 0;
		v[i]->t_pInf.z = 0;
		v[i]->r_pInf.x = 0;
		v[i]->r_pInf.y = 0;
		v[i]->r_pInf.z = 0;
		
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
	cart m2tom1 = m1->t_position - m2->t_position;
	// Get magnitude of the gravitational force
	double forceMag = gravForce(m1->mass, m2->mass, distance(m1, m2));
	// Influence m2 by that force
	m2tom1 = m2tom1*forceMag; 
	m2->t_addForce(m2tom1);
	
}

void gravPull(double uniMass, cart uniMassDist, Shape* s)
{
	// Get direction vector
	cart shapeToMass = uniMassDist - s->t_position;

	// Get magnitude of the gravitational force
	double forceMag = gravForce(uniMass, s->mass, distance(s->t_position, uniMassDist));

	// Influence m2 by that force 
	shapeToMass = shapeToMass*forceMag;
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

// Determines whether two shapes are moving towards each other
bool movingTowards(Shape* s1, Shape* s2)
{
	if(dotProd(s1->t_velocity, s2->t_velocity) < 0)
		return true;
	else
		return false;
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
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		for(int j=i; j<v.size(); j++){
			if(i != j && i < j){
				// Checks if they collide are moving towards each other
				/*if(collide(v[i], v[j]) && movingTowards(v[i], v[j])){
					resolveCollision(v[i], v[j], 0.9);
				}*/
				if(collide(v[i], v[j])){
					resolveCollisionSpring(v[i], v[j]);
				}

			}
		}
	}

}

void resolveCollision(Shape* s1, Shape* s2, double dampingConst)
{	
	cart c2toc1 = s1->t_position - s2->t_position;	
	cart c1toc2 = negate(c2toc1); 
	c2toc1 = normalize(c2toc1);
	c1toc2 = normalize(c1toc2);
	// Influence factor of s1 on s2
	cart t1 = normalize(s1->t_velocity); 
	double IFs1ons2 = dotProd(c1toc2, normalize(s1->t_velocity));
	// Influence factor of s2 on s1
	cart t2 = normalize(s2->t_velocity); 
	double IFs2ons1 = dotProd(c2toc1, normalize(s2->t_velocity));
	cart inf1 = s1->mass*IFs1ons2*s1->t_velocity;
	s1->t_addMomentum(negate(inf1));
	s2->t_addMomentum(inf1*dampingConst);

	cart inf2 = s2->mass*IFs2ons1*s2->t_velocity;
	s2->t_addMomentum(negate(inf2));
	s1->t_addMomentum(inf2*dampingConst);
}

void resolveCollisionSpring(Shape* s1, Shape* s2)
{
	// Find intrusion of s1 on s2
	double intrusion = s2->boundingSphere() - (distance(s1, s2) + s1->radius);
	// Force from the spring 
	double force = SPRING_CONST*intrusion; 
	// Direction of force
	cart c2toc1 = s1->t_position - s2->t_position;	
	c2toc1 = normalize(c2toc1);
	// Adding the two forces to the shapes
	s1->addForce(force*c2toc1);
	s2->addForce(force*negate(c2toc1));
}

void bounce(Shape* s, cart wall, double dampingConst)
{
	cart momentum = s->t_velocity*s->mass;
	cart momentumInf = multComponents(momentum, negate(wall));
	s->t_addMomentum(momentumInf*2*dampingConst);
}
	 	

///////////////////////
// Simulation Functions 
///////////////////////

void t_updateVel(double t, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		// Velocity change from gained/lossed momentum
		v[i]->t_velocity = v[i]->t_velocity + v[i]->t_pInf/v[i]->mass;
		// Velocity change from accelerations
		cart accel = v[i]->t_forces/v[i]->mass;
		v[i]->t_velocity = v[i]->t_velocity + accel*t;
	}
}

void r_updateVel(double t, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		// Velocity change from gained/lossed momentum
		cart incVel = divComponents(v[i]->r_pInf, v[i]->momentCM());
		v[i]->r_velocity = v[i]->r_velocity + incVel; 
		// Velocity change from accelerations
		cart accel = divComponents(v[i]->r_forces, v[i]->momentCM()); 
		v[i]->r_velocity = v[i]->r_velocity + accel*t;
	}
}

void t_updatePos(double t, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		double mass = v[i]->mass;

		cart vel = v[i]->t_velocity;

		cart accel = v[i]->t_forces/mass;

		// Using parabolic motion df = d0 + vt + 0.5at^2
		v[i]->t_position.x += vel.x*t + 0.5*accel.x*t*t; 
		v[i]->t_position.y += vel.y*t + 0.5*accel.y*t*t; 
		v[i]->t_position.z += vel.z*t + 0.5*accel.z*t*t; 
	}
}

// Move one timestep using the rotational forces (torques)  on all the objects
void r_updatePos(double t, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		double mass = v[i]->mass;

		cart rVel = v[i]->r_velocity;

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
	t_updateVel(t, v);
	r_updateVel(t, v); 
	t_updatePos(t, v);
	r_updatePos(t, v);
	// Reset force vectors
	resetForcesAndMomentums(v);
}

void wrapWorld(cart worldLimits, std::vector<Shape*> v)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<v.size(); i++){
		// Positive world limit breaches
		if(v[i]->t_position.x > worldLimits.x){
			// Original math
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
	// Physicall influences (gravity/magnetism etc)
	//gravAllShapes(v);

	// Universal gravity influence (earth etc)
	cart c = {0, -100, 0};
	gravAllMass(99999999999999, c, v);

	// Update position of all shapes
	advancePosAndReset(t, v);

	// Detect and resolve all collisions
	collideAndResolve(v);

	// If worldwrap is on, worldwrap all objects
	//cart lims = {100, 100, 100};
	//wrapWorld(lims, v);
	enforceBoundaries(v, physBoundaryMin, physBoundaryMax, 0.9);
}

void enforceBoundaries(std::vector<Shape*> s, cart min, cart max, double dampingConst)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i<s.size(); i++){
		// One for each boundary 
		if((s[i]->t_position.x + s[i]->boundingSphere()) > max.x)
			bounce(s[i], DIR_RIGHT, dampingConst);
		if((s[i]->t_position.y + s[i]->boundingSphere()) > max.y)
			bounce(s[i], DIR_UP, dampingConst);
		if((s[i]->t_position.z - s[i]->boundingSphere()) < max.z)
			bounce(s[i], DIR_FWRD, dampingConst);
		if((s[i]->t_position.x + s[i]->boundingSphere()) < min.x)
			bounce(s[i], DIR_LEFT, dampingConst);
		if((s[i]->t_position.y + s[i]->boundingSphere()) < min.y)
			bounce(s[i], DIR_DOWN, dampingConst);
		if((s[i]->t_position.z - s[i]->boundingSphere()) > min.z)
			bounce(s[i], DIR_BACK, dampingConst);
	}
}

void physicsThread(std::vector<Shape*> v)
{
	using namespace std::chrono;
	milliseconds sleepDur((unsigned int)(1/(SIM_FPS*2)));
	high_resolution_clock::time_point last = high_resolution_clock::now();
	high_resolution_clock::time_point now = high_resolution_clock::now();
	while(1){
		// check if enough time has passed
		now = high_resolution_clock::now();
		if(duration_cast<milliseconds>(now - last).count() >= 1000/SIM_FPS){
			//advanceSim((double)((duration_cast<std::chrono::milliseconds>(now - last).count())/((double)1000)), v);
			advanceSim(SIM_T, v);
			last = high_resolution_clock::now();
		}
	}
}
