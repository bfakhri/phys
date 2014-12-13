#include "mass.h"

Mass::Mass()
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

double Mass::getMass(){
	return objectMass; 
}

cartesian Mass::getPos(){
	return position; 
}

cartesian Mass::getVelocity(){
	return velocity; 
}

void Mass::setMass(double newMass){
	objectMass = newMass; 
}

void Mass::setPos(cartesian newPos){
	position = newPos;
}

void Mass::setVelocity(cartesian newVelocity){
	velocity = newVelocity; 
}

void Mass::resetForces(){
	cumalForces.x = 0; 
	cumalForces.y = 0; 
	cumalForces.z = 0; 
}

void Mass::addForce(cartesian force){
	cumalForces.x += force.x; 
	cumalForces.y += force.y; 
	cumalForces.z += force.z;
}	

double Mass::newtonGrav(double objMass, double distance){
	return G*(objectMass*objMass)/(distance*distance); 
}

void Mass::influence(Mass obj){
	cartesian objPos = obj.getPos();
	cartesian diffPos; 
	diffPos.x = position.x - objPos.x;
	diffPos.y = position.y - objPos.y;
	diffPos.z = position.z - objPos.z;

	double distance = sqrt(diffPos.x*diffPos.x + diffPos.y * diffPos.y + diffPos.z * diffPos.z); 

	double netForce = newtonGrav(obj.getMass(), distance); 

	// Check this math just in case
	cumalForces.x += netForce * diffPos.x/distance; 
	cumalForces.y += netForce * diffPos.y/distance; 
	cumalForces.z += netForce * diffPos.z/distance; 	
}
// MAKE FASTER BY GETTING RID OF REDUNDANT COMPUTING
// F = ma, a = F/m
// v = v0 + at  ---> v = v0 + F/m * t
// d = d0 + v0*t + 0.5*a*t^2
cartesian Mass::updateVelAndPos(double timeStep){
	// Updates the position
	position.x += velocity.x*timeStep + 0.5*(cumalForces.x/objectMass)*(timeStep*timeStep);
	position.x += velocity.y*timeStep + 0.5*(cumalForces.y/objectMass)*(timeStep*timeStep);
	position.x += velocity.z*timeStep + 0.5*(cumalForces.z/objectMass)*(timeStep*timeStep);
	
	// Updates the velocity
	velocity.x += (cumalForces.x/objectMass)*(timeStep); 
	velocity.y += (cumalForces.y/objectMass)*(timeStep); 
	velocity.z += (cumalForces.z/objectMass)*(timeStep); 
}
