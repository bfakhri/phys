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
#define GLOBAL_WORK_BUFF_SIZE 100
#define LOCAL_WORK_BUFF_SIZE 100

using namespace std; 

// NOTE - POSSIBLY SPEED THIS UP BY 
// CALLING IT USING OMP STRUCTURES
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
	
	// Output to make ensure OMP is working correctly on machine
	printf("Num Procs: %d\n\n", omp_get_num_procs()); 

	// START OF PARALLEL CODE
	#pragma omp parallel
	{
		double LOCAL_C_Arr[LOCAL_WORK_BUFF_SIZE];
		double LOCAL_D_Arr[LOCAL_WORK_BUFF_SIZE];
		int bufferCount = 0; 

		double localMax = 0; 

		do
		{
			// Check to see if we need work from above
			if(localEmpty)
			{
				bufferCount++;
				bufferPop(&LOCAL_C_Arr[bufferCount], &LOCAL_D_Arr[bufferCount]);				
			}
			// Update the local Max
			// REMEMBER TO INCLUDE EPSILON
			if(GLOB_MAX > localMax)
				localMax = GLOB_MAX;
			bool isChanged = false; 
			if(localMax < f(LOCAL_C_Arr[bufferCount]))
			{
				localMax = f(LOCAL_C_Arr[bufferCount]);
				isChanged = true; 
			}
			if(localMax < f(LOCAL_D_Arr[bufferCount]))
			{
				localMax = f(LOCAL_D_Arr[bufferCount]);
				isChanged = true; 
			}

			if(isChanged)
			{
				// We have a new max so we have to pursue this interval
				
			}
			else
			{
				// We don't have a new max, throw away this interval
				// Check if it is possible to have larger value inside using slope 
				if(*possible*)
				{
					// Add new intervals to the buffer
					// Add to buffer
					if(localBufferFull)
					{
						// Add to global buffer
					}
					else
					{
						// Add to local buffer
					}

					// throw away current interval
				
				}
				// Throw away interval
				else
				{

				}
			}
			// REMEMBER TO USE EPSILON
			// SAFEGUARD THIS WITH MUTEX
			if(localMax > GLOB_Max)
			{
				GLOB_Max = localMax;
			}
	
		}
		// Keep going while the local buffer isn't empty OR the global buffer isn't empty
		while(*localNotEmpty* || *globalNotEmpty*); 
		

		printf("My ThreadNum: %d\n\n", myNum); 
	}
	return 0; 
}
