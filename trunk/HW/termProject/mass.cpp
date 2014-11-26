#include "mass.h"


double mass::getMass(){
	return mass; 
}

cartesian mass::getPos(){
	return position; 
}

cartesian mass::getVelocity(){
	return velocity; 
}

void mass::setMass(double newMass){
	mass = newMass; 
}

void mass::setPos(cartesian newPos){
	position = newPos;
}

void mass::setVelocity(cartestian newVelocity){
	velocity = newVelocity; 
}

void mass::resetForces(){
	cumalForces.x = 0; 
	cumalForces.y = 0; 
	cumalForces.z = 0; 
}

void mass::addForce(cartesian force){
	cumalForces.x += force.x; 
	cumalForces.y += force.y; 
	cumalForces.z += force.z;
}	

// MAKE FASTER BY GETTING RID OF REDUNDANT COMPUTING
// F = ma, a = F/m
// v = v0 + at  ---> v = v0 + F/m * t
// d = d0 + v0*t + 0.5*a*t^2
cartesian mass::updateVelAndPos(double timeStep){
	// Updates the position
	position.x += velocity.x*timeStep + 0.5*(cumalForces.x/mass)*(timeStep*timeStep);
	position.x += velocity.y*timeStep + 0.5*(cumalForces.y/mass)*(timeStep*timeStep);
	position.x += velocity.z*timeStep + 0.5*(cumalForces.z/mass)*(timeStep*timeStep);
	
	// Updates the velocity
	velocity.x += (cumalForces.x/mass)*(timeStep); 
	velocity.y += (cumalForces.y/mass)*(timeStep); 
	velocity.z += (cumalForces.z/mass)*(timeStep); 
}