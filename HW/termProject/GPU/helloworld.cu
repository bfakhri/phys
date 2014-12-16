#include <stdio.h>
#include <iostream>
#include "mass.cu"

const int N = 7;


__global__
void simulate(Mass * masses, unsigned long numMasses, double deltaT, unsigned long totalTimeSteps)
{
	for(unsigned long i=0; i<totalTimeSteps; i++)
	{
		// Sync threads so positions are not updated before all other 
		__syncthreads(); 

		// Calc forces on all masses
		for(unsigned long i=0; i<numMasses; i++)
		{
			if(i != threadIdx.x)
			{
				influence(&masses[threadIdx.x], &masses[i]); 
			}
		}

		// Sync threads so positions are not updated before all other 
		__syncthreads(); 

		// Update position of all masses
		updateVelAndPos(&masses[threadIdx.x], deltaT); 

		// Reset forces
		resetForces(&masses[threadIdx.x]);
	} 
}


int main(int argc, char ** argv)
{
	// Simulation parameter variables
	double TIME_STEP_SIZE = 1;
	unsigned long TOTAL_SIM_STEPS = 1000;

	// Custom simulation parameters 
	if(argc > 1)
	{
		if(argc != 3){
			std::cout << std::endl << "ERROR, incorrect number of arguments" << std::endl; 
			return -1; 
		}else{
			TOTAL_SIM_STEPS = atoi(argv[1]); 
			TIME_STEP_SIZE = atoi(argv[2]); 
		}
	}
	std::cout << "Simulation: " << std::endl << "Number of steps = " << TOTAL_SIM_STEPS 
		<< std::endl << "Size of time step (seconds) = " << TIME_STEP_SIZE << std::endl;
 

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


	// Allocate memory on device for masses
	Mass * d_massArray;
	cudaMalloc( (void**)&d_massArray, N*sizeof(Mass));

	// Copy masses onto device
	cudaMemcpy( d_massArray, h_massArray, (N*sizeof(Mass)), cudaMemcpyHostToDevice );

	// Dimensions for cuda function call 
	dim3 blockDimensions( N, 1 );
	dim3 gridDimensions( 1, 1 );

	// Do sim
	simulate<<< gridDimensions, blockDimensions >>>(d_massArray, N, TIME_STEP_SIZE, TOTAL_SIM_STEPS);

	// Get data back
	cudaMemcpy( h_massArray, d_massArray, (N*sizeof(Mass)), cudaMemcpyDeviceToHost );

	// Free device mem 
	cudaFree( d_massArray );

	// Output
	// Some output

	return EXIT_SUCCESS;
}
