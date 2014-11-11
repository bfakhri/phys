// Tentative questions: 
// In email, when describing f(x), the 3rd line is "z = x". Why isn't it z = 0? 
// Try setting number of threads to 1.5X number of procs to do hyperthreading? 

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#define START_A 1
#define END_B 100
#define epsilon 0.000001 // 10^-6
#define slope 12
#define WORK_BUFF_SIZE 100

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
double GLOB_C_Arr[WORK_BUFF_SIZE];
double GLOB_D_Arr[WORK_BUFF_SIZE];
bool GLOB_BuffFull = false;	// Signals whether work buffer is full

// Maybe hide these variable better? 
int buffHead = 0; 
int buffTail = 1; 

// These buffer manipulation functions
// should only be allowed to be called 
// on thread at a time!
// Check to make sure negatives work for the head/tail with modulo
void bufferPush(double c, double d)
{
	GLOB_C_Arr[buffTail] = c; 
	GLOB_D_Arr[buffTail] = d; 
	buffTail = (buffTail+1)%WORK_BUFF_SIZE; 
	if(buffHead == buffTail)
	{
		// Buffer is now full
		GLOB_BuffFull = true; 
	}
}

void bufferPop(double* c, double* d)
{
	*c = GLOB_C_Arr[buffHead];
	*d = GLOB_D_Arr[buffHead];
	buffHead = (buffHead-1)%WORK_BUFF_SIZE; 

	// THIS IS NOT THE RIGHT INDICATOR FOR EMPTINESS
	if(buffHead == buffTail)
	{
		// Buffer is now empty
		GLOB_BuffFull = false; 
	}
}


int main(int argc, char *argv[]) 
{
	omp_set_num_threads(omp_get_num_procs()); 
	double input = 0; 
	input = atoi(argv[1]); 
	// This is just to test f
	printf("f(%2.0f) = %1.13f\n\n", input, f(input)); 
	printf("Num Procs: %d\n\n", omp_get_num_procs()); 
	printf("My ThreadNum: %d\n\n", omp_get_thread_num()); 
	
	return 0; 
}
