#include "mass.h"

mass::mass()
{
	objectMass = 1; 	// Default 1Kg
	position.x = 0; 
	position.y = 0; 
	position.z = 0; 

	velocity.x = 0; 
	velocity.y = 0; 
	velocity.z = 0; 

	cumalForces.x = 0; 
	cumalForces.y = 0; 
	cumalForces.z = 0; 
		
}

double mass::getMass(){
	return objectMass; 
}

cartesian mass::getPos(){
	return position; 
}

cartesian mass::getVelocity(){
	return velocity; 
}

void mass::setMass(double newMass){
	objectMass = newMass; 
}

void mass::setPos(cartesian newPos){
	position = newPos;
}

void mass::setVelocity(cartesian newVelocity){
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
	position.x += velocity.x*timeStep + 0.5*(cumalForces.x/objectMass)*(timeStep*timeStep);
	position.x += velocity.y*timeStep + 0.5*(cumalForces.y/objectMass)*(timeStep*timeStep);
	position.x += velocity.z*timeStep + 0.5*(cumalForces.z/objectMass)*(timeStep*timeStep);
	
	// Updates the velocity
	velocity.x += (cumalForces.x/objectMass)*(timeStep); 
	velocity.y += (cumalForces.y/objectMass)*(timeStep); 
	velocity.z += (cumalForces.z/objectMass)*(timeStep); 
}
