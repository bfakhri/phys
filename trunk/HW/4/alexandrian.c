#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#define START_A 1
#define END_B 1000
#define epsilon 0.000000000001 // 10^-12
#define slope 12

using namespace std; 

// NOTE - POSSIBLY SPEED THIS UP BY SPLITTING ITERATIONS
double mathFunction(double x)
{
	double outerSum = 0; 
	for(unsigned int i = 100; i >= 1; --i)
	{
		double innerSum = 0; 
		for(unsigned int j = i; j >= i; --j)
		{
			innerSum += pow((x + j), -3.1);
		}
		
		outerSum += sin(x + innerSum)/pow(1.2, i);
	}

	return outerSum; 
}


double GLOB_MAX = 0; 

int main(int argc, char *argv[]) {

	printf("Num Procs: %d\n\n", omp_get_num_procs()); 

	omp_set_num_threads(omp_get_num_procs()); 
	omp_get_thread_num(); 
	return 0; 
}
