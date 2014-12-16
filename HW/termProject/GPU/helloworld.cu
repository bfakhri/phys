#include <stdio.h>
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
			if(i != threadIx.x)
			{
				influence(masses[threadIx.x], masses[i]); 
			}
		}

		// Sync threads so positions are not updated before all other 
		__syncthreads(); 

		// Update position of all masses
		updateVelAndPos(masses[threadIx.x], timeStep); 

		// Reset forces
		resetForces(masses[threadIx.x]);
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
			cout << endl << "ERROR, incorrect number of arguments" << endl; 
			return -1; 
		}else{
			TOTAL_SIM_STEPS = atoi(argv[1]); 
			TIME_STEP_SIZE = atoi(argv[2]); 
		}
	}
	cout << "Simulation: " << endl << "Number of steps = " << SIM_STEPS 
		<< endl << "Size of time step (seconds) = " << TIME_STEP << endl;
 
	const int massArraySize = N*sizeof(Mass);

	cudaMalloc( (void**)&ad, csize );
	cudaMalloc( (void**)&bd, isize );
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

	dim3 blockDimensions( N, 1 );
	dim3 gridDimensions( 1, 1 );

	simulate<<< gridDimensions, blockDimensions >>>(massArray, N, stepSize, totalSteps);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
	cudaFree( ad );

	printf("%s\n", a);
	return EXIT_SUCCESS;
}
