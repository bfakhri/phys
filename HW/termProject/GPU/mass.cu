double scientificNotation(double num,  int exp)
{ 
	if(exp > 0)
	{
		for(int i=0; i<exp; i++){
			num *= 10; 
		}
	}else{
		for(int i=0; i>exp; i--){
			num /= 10; 
		}
	}
	return num; 
}

double G = scientificNotation(6.67384, -11);

typedef struct Mass
{
	double objectMass;
	double positionX;
	double positionY;
	double positionZ;
	double velocityX;
	double velocityY;
	double velocityZ;
	double cumalForcesX;
	double cumalForcesY;
	double cumalForcesZ;
}Mass;

__device__
void resetForces(Mass *m){
	m->cumalForcesX = 0;
	m->cumalForcesY = 0;
	m->cumalForcesZ = 0;
}

__device__
double newtonGrav(double m1Mass, double m2Mass, double distance){
	return G*(m1Mass*m2Mass)/(distance*distance);
}

__device__
void influence(Mass *m1, Mass *m2){
	double diffPosX = m1->positionX - m2->positionX;
	double diffPosY = m1->positionY - m2->positionY;
	double diffPosZ = m1->positionZ - m2->positionZ;
	double distance = sqrt(diffPosX*diffPosX + diffPosY*diffPosY + diffPosZ*diffPosZ);
	double netForce = newtonGrav(m1->objectMass, m2->objectMass, distance);
	m1->cumalForcesX += netForce * diffPosX/distance; 
	m1->cumalForcesY += netForce * diffPosY/distance; 
	m1->cumalForcesZ += netForce * diffPosZ/distance;
}

__device__
void updateVelAndPos(Mass *m, double timeStep){
	double accelerationX = m->cumalForcesX/m->objectMass;
	double accelerationY = m->cumalForcesY/m->objectMass;
	double accelerationZ = m->cumalForcesZ/m->objectMass;
	double timeStepSquared = timeStep*timeStep;

	m->positionX += m->velocityX*timeStep + 0.5*(accelerationX)*(timeStepSquared);
	m->positionY += m->velocityY*timeStep + 0.5*(accelerationY)*(timeStepSquared);
	m->positionZ += m->velocityZ*timeStep + 0.5*(accelerationZ)*(timeStepSquared);

	m->velocityX += (accelerationX)*(timeStep);
	m->velocityY += (accelerationY)*(timeStep);
	m->velocityZ += (accelerationZ)*(timeStep);
}
