


// Tentative questions: 
// In email, when describing f(x), the 3rd line is "z = x". Why isn't it z = 0? 
// Try setting number of threads to 1.5X number of procs to do hyperthreading? 

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#define START_A 1
#define END_B 100
#define EPSILON 0.000001 // 10^-6
#define SLOPE 12
#define GLOBAL_BUFF_SIZE 100
#define LOCAL_BUFF_SIZE 100

#define STATUS_EMPTY -1
#define STATUS_MID 0
#define STATUS_FULL 1


using namespace std; 

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


// Local Circular Queue
bool local_qWork(double c, double d, double * buffer, int * head, int * tail, int * status)
{
	if(*status == STATUS_FULL)
	{
		return false;
	}
	else
	{
		// Add to circular buffer
		buffer[*tail] = c;
		buffer[*tail+1] = d; 
		*tail = (*tail+2)%LOCAL_BUFF_SIZE;  
		if(*tail == *head)
			*status = STATUS_FULL;
		else
			*status = STATUS_MID; 

		return true; 
	}
}

bool local_deqWork(double * c, double * d, double * buffer, int * head, int * tail, int * status)
{
	if(*status == STATUS_EMPTY)
	{
		return false;
	}
	else
	{
		// Add to circular buffer
		*c = buffer[*head];
		*d = buffer[*head+1]; 
		*head = (*head+2)%LOCAL_BUFF_SIZE;  
		if(*tail == *head)
			*status = STATUS_EMPTY;
		else
			*status = STATUS_MID; 

		return true; 
	}
}

// Returns true only if max changed
bool local_setMax(double * currentMax, double fc, double fd)
{
	if(*currentMax + EPSILON < fc)
		*currentMax = fc;
	else if(*currentMax + EPSILON < fd)
		*currentMax = fd; 
	else 
		return false; 

	return true; 
}

// Returns true only if it is possible to get a higher value in this interval
bool validInterval(double currenMax, double c, double d)
{
	if(((f(c) + f(d) + SLOPE*(d - c))/2) > (currentMax + EPSILON))
		return true; 
	else
		return false;
}

int main()
{
	// Init local variables
	double local_max = 0; 
	double local_buffer[LOCAL_BUFF_SIZE]; 
	double local_c = 0;
	double local_d = 0; 
	int local_head = 0; 
	int local_tail = 0; 
	int local_status = STATUS_EMPTY; 

	// Add init interval to queue
	local_qWork(START_A, END_B, local_buffer, &local_head, &local_tail, &local_status);


	do
	{
		// Get work from queue
		local_deqWork(&local_c, &local_d, local_buffer, &local_head, &local_tail, &local_status);

		// Maybe reorganize to change max first? 
		// Check if possible larger
		if(validInterval(currentMax, local_c, local_d))
		{
			// Maybe set max somewhere else for higher efficiency? 
			// Use the boolean output maybe? 
			local_setMax(*local_max, f(local_c), f(local_d)); 
			// Determine whether all of these are necessary 
			local_qWork(local_c, local
				
		}
		
	}while(stat != STATUS_EMPTY); 

	return 0; 	 
}
