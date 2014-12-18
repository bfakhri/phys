#include <stdio.h>
#include <iostream>
#include <math.h>
#include <omp.h> 
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

__host__
__device__
void resetForces(Mass *m){
	m->cumalForcesX = 0;
	m->cumalForcesY = 0;
	m->cumalForcesZ = 0;
}

__host__
__device__
double newtonGrav(double m1Mass, double m2Mass, double distance, double localG){
	return localG*(m1Mass*m2Mass)/(distance*distance);
}

__host__
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

__host__
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

__host__
inline void h_simulateTic(Mass * masses, unsigned long numMasses, double deltaT, Mass proxyMass, double localG)
{
	// Calc forces on all masses
	#pragma omp parallel for schedule(static)
	for(unsigned long i=0; i<numMasses; i++){ 
		for(unsigned long j=0; j<numMasses; j++){
			if(i != j)
				influence(&masses[i], &masses[j], localG); 
		}
		influence(&masses[i], &proxyMass, localG); 
	}

	// Update position of all masses
	#pragma omp parallel for schedule(static)
	for(unsigned int i=0; i<numMasses; i++){
		updateVelAndPos(&masses[i], deltaT); 
		resetForces(&masses[i]);
	}
	 
}
__global__
void d_simulateTic(Mass * masses, unsigned long numMasses, unsigned int massesPerBlock, double deltaT, Mass * proxyMass, double localG)
{
	unsigned int myId = blockIdx.x*massesPerBlock + threadIdx.x; 
	// Calc forces on all masses
	for(unsigned long i=0; i<numMasses; i++){
		if(i != myId)
			influence(&masses[myId], &masses[i], localG); 
	}
	influence(&masses[myId], proxyMass, localG); 

	// Sync threads so positions are not updated before all other 
	__syncthreads(); 

	// Update position of all masses
	updateVelAndPos(&masses[myId], deltaT); 

	// Reset forces
	resetForces(&masses[myId]);
	 
}

__global__
void d_computeProxyMass(Mass * masses, Mass * proxyMass, unsigned int numMasses)
{
	proxyMass->objectMass = 0; 
	proxyMass->positionX = 0; 
	proxyMass->positionY = 0; 
	proxyMass->positionZ = 0;
	for(unsigned int i=0; i<numMasses; i++){
		proxyMass->objectMass += masses[i].objectMass; 
		proxyMass->positionX += masses[i].positionX*masses[i].objectMass; 
		proxyMass->positionY += masses[i].positionY*masses[i].objectMass; 
		proxyMass->positionZ += masses[i].positionZ*masses[i].objectMass; 
	}
	proxyMass->positionX /= proxyMass->objectMass; 
	proxyMass->positionY /= proxyMass->objectMass; 
	proxyMass->positionZ /= proxyMass->objectMass; 
}
__host__
void h_computeProxyMass(Mass * masses, Mass * proxyMass, unsigned int numMasses)
{
	proxyMass->objectMass = 0; 
	proxyMass->positionX = 0; 
	proxyMass->positionY = 0; 
	proxyMass->positionZ = 0;
	#pragma omp parallel for schedule(static)
	for(unsigned int i=0; i<numMasses; i++){
		proxyMass->objectMass += masses[i].objectMass; 
		proxyMass->positionX += masses[i].positionX*masses[i].objectMass; 
		proxyMass->positionY += masses[i].positionY*masses[i].objectMass; 
		proxyMass->positionZ += masses[i].positionZ*masses[i].objectMass; 
	}
	proxyMass->positionX /= proxyMass->objectMass; 
	proxyMass->positionY /= proxyMass->objectMass; 
	proxyMass->positionZ /= proxyMass->objectMass; 
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

	unsigned int L = 1536;//512;//N/2; 	// Number of masses on CPU
	unsigned int M = 512;//N/2;	// Number of masses on GPU 
	// Custom simulation parameters 
	if(argc > 1)
	{
		if(argc != 6){
			std::cout << std::endl << "ERROR, incorrect number of arguments" << std::endl; 
			return -1; 
		}else{
			TOTAL_SIM_STEPS = atoi(argv[1]); 
			TIME_STEP_SIZE = atoi(argv[2]); 
			N = atoi(argv[3]); 
			//MASSES_PER_CORE = atoi(argv[4]); 
			L = atoi(argv[4]);
			M = atoi(argv[5]); 
		}
	}else
	{
		TOTAL_SIM_STEPS = 100; 
		TIME_STEP_SIZE = 10000; 
		N = 256; 
		MASSES_PER_CORE = 1; 

	}
	std::cout << "Simulation: " << std::endl << "Number of steps = " << TOTAL_SIM_STEPS 
		<< std::endl << "Size of time step (seconds) = " << TIME_STEP_SIZE
		<< std::endl << "N (number of masses) = " << N
		<< std::endl << "Masses per core = " << MASSES_PER_CORE 
		<< std::endl << "Value of G = " << G << std::endl;

 

	// Make masses	
	Mass * h_massArray =(Mass*) malloc(L*sizeof(Mass)); 
	Mass * tmp_massArray =(Mass*) malloc(M*sizeof(Mass)); 
	
	// Populate host array of masses
	for(int i=0; i<L; i++){
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

	for(int i=L; i<(M+L); i++){
		tmp_massArray[i-L].objectMass = scientificNotation(6+i, 23); 
		tmp_massArray[i-L].positionX = scientificNotation(6+i, 10); 	
		tmp_massArray[i-L].positionY = scientificNotation(6+i, 10); 	
		tmp_massArray[i-L].positionZ = scientificNotation(6+i, 10); 	
		tmp_massArray[i-L].velocityX = scientificNotation(6+i, 1); 	
		tmp_massArray[i-L].velocityY = scientificNotation(6+i, 2); 	
		tmp_massArray[i-L].velocityZ = scientificNotation(6+i, 1); 	
		tmp_massArray[i-L].cumalForcesX = 0; 	
		tmp_massArray[i-L].cumalForcesY = 0; 	
		tmp_massArray[i-L].cumalForcesZ = 0; 	
	}

	// Start output
	std::cout << "Start Posisions (X): " << std::endl;
	std::cout << h_massArray[0].positionX << std::endl; 
	std::cout << tmp_massArray[0].positionX << std::endl;


	// Allocate memory on device for masses
	std::cout << "M: " << M << std::endl;
	Mass * d_massArray;
	cudaMalloc( (void**)&d_massArray, M*sizeof(Mass));
	//Mass * d_massArray;
	//cudaMalloc( (void**)&d_massArray, N*sizeof(Mass));
	
	Mass * d_proxyMassCPU;
	cudaMalloc( (void**)&d_proxyMassCPU, sizeof(Mass));
	Mass * d_proxyMassGPU;
	cudaMalloc( (void**)&d_proxyMassGPU, sizeof(Mass));
	
	// Copy masses onto device
	cudaMemcpy( d_massArray, tmp_massArray, (M*sizeof(Mass)), cudaMemcpyHostToDevice );

	// Free local memory for temp array
	free(tmp_massArray); 

	// Dimensions for cuda function call 
	dim3 blockDimensions( 32, 1 );
	dim3 gridDimensions( 8, 1 );
	dim3 blockSingle(1, 1); 
	dim3 gridSingle(1, 1); 	

	// Do simi
	unsigned int massesPerBlock = M/blockDimensions.x; 


	// Split data between CPU and GPU
	// Masses 0-(dividor-1) go to CPU
	// Masses dividor-(N-1) go to GPU	
	
	for(unsigned int t=0; t<TOTAL_SIM_STEPS; t++)
	{
		// Make proxy mass for GPU masses
		d_computeProxyMass<<< gridSingle, blockSingle >>>(d_massArray, d_proxyMassGPU, L); 

		// Make proxy mass for CPU masses
		Mass h_proxyMassCPU;
		h_computeProxyMass(h_massArray, &h_proxyMassCPU, L); 
	
		// Send CPU proxy mass to GPU
		cudaMemcpy( d_proxyMassCPU, &h_proxyMassCPU, sizeof(Mass), cudaMemcpyHostToDevice );
		
		// Receive proxy mass from GPU 
		Mass h_proxyMassGPU;
		cudaMemcpy( &h_proxyMassGPU, d_proxyMassGPU, sizeof(Mass), cudaMemcpyDeviceToHost );
	
		// Perform one tick of simulation on CPU and GPU 
		// CALL GPU FUNCT
		d_simulateTic <<< gridDimensions, blockDimensions >>>(d_massArray, M, massesPerBlock, TIME_STEP_SIZE, d_proxyMassCPU, G); 
		// CALL CPU FUNCT
		h_simulateTic(h_massArray, L, TIME_STEP_SIZE, h_proxyMassGPU, G); 
	}
 

	tmp_massArray =(Mass*) malloc(M*sizeof(Mass)); 
	cudaMemcpy( tmp_massArray, d_massArray, (M*sizeof(Mass)), cudaMemcpyDeviceToHost );

	// Output
	std::cout << std::endl << "End Posisions (X): " << std::endl;
	std::cout << h_massArray[0].positionX << std::endl; 
	std::cout << tmp_massArray[0].positionX << std::endl;
	
	return EXIT_SUCCESS;
}
