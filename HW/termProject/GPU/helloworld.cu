// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include "mass.cu"

const int N = 7;
const int blocksize = 7;

__global__
void hello(char *a, int *b)
{
 a[threadIdx.x] += b[threadIdx.x];
}

__global__
void simulateOneTick(Mass * masses, unsigned long numMasses, double deltaT)
{
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

__global__
void simulateOnDevice(Mass * masses, unsigned long numMasses, double deltaT, unsigned long totalTimeSteps)
{
	// Calc forces on all masses
	for(unsigned long i=0; i<numMasses; i++)
	{
		// Perform step
		simulateOneTick(masses, numMasses, deltaT); 
		// Sync threads
		// **** IS THIS AUTOMATICALLY SYNCED!? **** // 
		
	}
}

int main()
{
 char a[N] = "Hello ";
 int b[N] = {15, 10, 6, 0, -11, 1, 0};

 char *ad;
 int *bd;
 const int csize = N*sizeof(char);
 const int isize = N*sizeof(int);

 printf("%s", a);

 cudaMalloc( (void**)&ad, csize );
 cudaMalloc( (void**)&bd, isize );
 cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
 cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

 dim3 dimBlock( blocksize, 1 );
 dim3 dimGrid( 1, 1 );
 hello<<<dimGrid, dimBlock>>>(ad, bd);
 cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost );
 cudaFree( ad );

 printf("%s\n", a);
 return EXIT_SUCCESS;
}
