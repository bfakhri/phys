#include <stdio.h>
#include <iostream>
#include <math.h>
//#include "mass.cu"

unsigned int N;
unsigned int MASSES_PER_CORE;

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

double G = 0;

void initG(){
	G = scientificNotation(6.67384, -11);
}

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
double newtonGrav(double m1Mass, double m2Mass, double distance, double localG){
	return localG*(m1Mass*m2Mass)/(distance*distance);
}

__device__
void influence(Mass *m1, Mass *m2, double localG){
	double diffPosX = m1->positionX - m2->positionX;
	double diffPosY = m1->positionY - m2->positionY;
	double diffPosZ = m1->positionZ - m2->positionZ;
	double distance = sqrt(diffPosX*diffPosX + diffPosY*diffPosY + diffPosZ*diffPosZ);
	//*dist = distance; 
	//*dist = 15; 
	double netForce = newtonGrav(m1->objectMass, m2->objectMass, distance, localG);
	//*dist = netForce;
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
	//*dist = m->velocityX*timeStep + 0.5*(accelerationX)*(timeStepSquared);
	m->positionY += m->velocityY*timeStep + 0.5*(accelerationY)*(timeStepSquared);
	m->positionZ += m->velocityZ*timeStep + 0.5*(accelerationZ)*(timeStepSquared);

	m->velocityX += (accelerationX)*(timeStep);
	m->velocityY += (accelerationY)*(timeStep);
	m->velocityZ += (accelerationZ)*(timeStep);
}

__global__
void simulate(Mass * masses, unsigned long numMasses, unsigned int massesPerBlock, double deltaT, unsigned long totalTimeSteps, double localG)
{
	unsigned int myId = blockIdx.x*massesPerBlock + threadIdx.x; 
	for(unsigned int i=0; i<totalTimeSteps; i++)
	{
		// Sync threads so positions are not updated before all other 
		__syncthreads(); 

		// Calc forces on all masses
		for(unsigned long i=0; i<numMasses; i++){
			if(i != myId)
				influence(&masses[myId], &masses[i], localG); 
		}
		

		// Sync threads so positions are not updated before all other 
		__syncthreads(); 

		// Update position of all masses
		updateVelAndPos(&masses[myId], deltaT); 

		// Reset forces
		resetForces(&masses[myId]);
	} 
}

__global__
void testEff(Mass * masses)//, unsigned long numMasses, double deltaT, unsigned long totalTimeSteps, double localG)
{
	masses[threadIdx.x].positionX = (double)threadIdx.x; 
}

__global__
void testInfluence(Mass * masses, unsigned int numMasses, double localG){
	for(unsigned long i=0; i<numMasses; i++)
	{
		if(i != threadIdx.x)
		{
			influence(&masses[threadIdx.x], &masses[i], localG); 
		}
	}
}


int main(int argc, char ** argv)
{
	initG(); 

	// Simulation parameter variables
	double TIME_STEP_SIZE = 1;
	unsigned long TOTAL_SIM_STEPS = 1000;

	// Custom simulation parameters 
	if(argc > 1)
	{
		if(argc != 5){
			std::cout << std::endl << "ERROR, incorrect number of arguments" << std::endl; 
			return -1; 
		}else{
			TOTAL_SIM_STEPS = atoi(argv[1]); 
			TIME_STEP_SIZE = atoi(argv[2]); 
			N = atoi(argv[3]); 
			MASSES_PER_CORE = atoi(argv[4]); 
		}
	}
	std::cout << "Simulation: " << std::endl << "Number of steps = " << TOTAL_SIM_STEPS 
		<< std::endl << "Size of time step (seconds) = " << TIME_STEP_SIZE
		<< std::endl << "N (number of masses) = " << N
		<< std::endl << "Masses per core = " << MASSES_PER_CORE 
		<< std::endl << "Value of G = " << G << std::endl;
 

	// Make masses	
	Mass * h_massArray =(Mass*) malloc(N*sizeof(Mass)); 
	
	// Populate array of masses
	for(int i=0; i<N; i++){
		h_massArray[i].objectMass = scientificNotation(6+i, 23); 
		h_massArray[i].positionX = scientificNotation(6+i, 10); 	
		h_massArray[i].positionY = scientificNotation(6+i, 10); 	
		h_massArray[i].positionZ = scientificNotation(6+i, 10); 	
		h_massArray[i].velocityX = scientificNotation(6+i, 1); 	
		h_massArray[i].velocityY = scientificNotation(6+i, 2); 	
		h_massArray[i].velocityZ = scientificNotation(6+i, 1); 	
		h_massArray[i].cumalForcesX = 0; 	
		h_massArray[i].cumalForcesY = 0; 	
		h_massArray[i].cumalForcesZ = 0; 	
	}

	// Start output
	std::cout << "Start Posisions (X): " << std::endl;
	//for(int i=0; i<N; i++){
	//	std::cout << h_massArray[i].positionX << std::endl; 
	//}
	std::cout << h_massArray[0].positionX << std::endl; 


	// Allocate memory on device for masses
	Mass * d_massArray;
	cudaMalloc( (void**)&d_massArray, N*sizeof(Mass));


	// Copy masses onto device
	cudaMemcpy( d_massArray, h_massArray, (N*sizeof(Mass)), cudaMemcpyHostToDevice );

	// Dimensions for cuda function call 
	dim3 blockDimensions( 32, 1 );
	dim3 gridDimensions( 8, 1 );

	// Do simi
	unsigned int massesPerBlock = N/blockDimensions.x; 
	//unsigned int massesPerThread = massesPerBlock/gridDimensions.x;


	// Split data between CPU and GPU
	
	
	for(unsigned int t=0; t<TOTAL_TIME_STEPS; t++)
	{
		// Make proxy mass for both clumps
		Mass proxyMassCPU;
		proxyMassCPU.objectMass = 0; 
		proxyMassCPU.positionX = 0; 
		proxyMassCPU.positionY = 0; 
		proxyMassCPU.positionZ = 0; 
		for(unsigned int d=0; d<dividor; d++)
		{
			proxyMassCPU.objectMass += h_massArray[d].objectMass;
			proxyMassCPU.positionX += h_massArray[d].positionX;
			proxyMassCPU.positionY += h_massArray[d].positionY;
			proxyMassCPU.positionZ += h_massArray[d].positionZ;
		}
		// Calculate averages
		proxyMassCPU.objectMass /= dividor;
		proxyMassCPU.positionX /= dividor;
		proxyMassCPU.positionY /= dividor;
		proxyMassCPU.positionZ /= dividor;
		
			
		Mass proxyMassGPU;
		proxyMassGPU.objectMass = 0; 
		proxyMassGPU.positionX = 0; 
		proxyMassGPU.positionY = 0; 
		proxyMassGPU.positionZ = 0; 
		for(unsigned int d=0; d<N-dividor; d++)
		{
			proxyMassGPU.objectMass += h_massArray[d+dividor].objectMass;
			proxyMassGPU.positionX += h_massArray[d+dividor].positionX;
			proxyMassGPU.positionY += h_massArray[d+dividor].positionY;
			proxyMassGPU.positionZ += h_massArray[d+dividor].positionZ;
		}
		// Calculate averages
		proxyMassGPU.objectMass /= (N-dividor);
		proxyMassGPU.positionX /= (N-dividor);
		proxyMassGPU.positionY /= (N-dividor);
		proxyMassGPU.positionZ /= (N-dividor);
		
		// Send GPU its portion of the array and proxy mass

		// Perform one tick of simulation


	}
 
	simulate<<< gridDimensions, blockDimensions >>>(d_massArray, N, massesPerBlock, TIME_STEP_SIZE, TOTAL_SIM_STEPS, G);
	//simulate<<< gridDimensions, blockDimensions >>>(d_massArray, N, MASSES_PER_CORE, TIME_STEP_SIZE, TOTAL_SIM_STEPS, G);
	//testEff<<< gridDimensions, blockDimensions >>>(d_massArray);
	//testInfluence<<< gridDimensions, blockDimensions >>>(d_massArray, N, G, d_dist); 

	// Get data back
	cudaMemcpy( h_massArray, d_massArray, (N*sizeof(Mass)), cudaMemcpyDeviceToHost );

	// Free device mem 
	cudaFree( d_massArray );

	// Output
	std::cout << std::endl << "End Posisions (X): " << std::endl;
	//std::cout << "Distance: " << h_dist << std::endl;
	//for(int i=0; i<N; i++){
	//	std::cout << h_massArray2[i].positionX << std::endl; 
	//	std::cout << h_massArray2[i].cumalForcesX << std::endl; 	
	//}
	std::cout << h_massArray[0].positionX << std::endl; 

	return EXIT_SUCCESS;
}
