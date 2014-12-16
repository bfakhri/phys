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
