// Tentative questions: 
// In email, when describing f(x), the 3rd line is "z = x". Why isn't it z = 0? 
//

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#define START_A 1
#define END_B 100
#define epsilon 0.000001 // 10^-6
#define slope 12

using namespace std; 

// NOTE - POSSIBLY SPEED THIS UP BY SPLITTING ITERATIONS
double f(double x)
{
	double outerSum = 0; 
	for(unsigned int i = 100; i >= 1; --i)
	{
		double innerSum = 0; 
		for(unsigned int j = i; j >= 1; --j)
		{
			innerSum += pow((x + j), -3.1);
		}
		
		outerSum += sin(x + innerSum)/pow(1.2, i);
	}

	return outerSum; 
}


double GLOB_MAX = 0; 

int main(int argc, char *argv[]) 
{
	double input = 0; 
	input = atoi(argv[1]); 
	// This is just to test f
	printf("f(%2.0f) = %1.13f\n\n", input, f(input)); 
	printf("Num Procs: %d\n\n", omp_get_num_procs()); 
	
	omp_set_num_threads(omp_get_num_procs()); 
	omp_get_thread_num(); 
	return 0; 
}
