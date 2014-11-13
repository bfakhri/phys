// Tentative questions: 
// In email, when describing f(x), the 3rd line is "z = x". Why isn't it z = 0? 
// Try setting number of threads to 1.5X number of procs to do hyperthreading? 



#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <iostream>

#define START_A 1
#define END_B 5 
#define EPSILON 0.000001 // 10^-6
#define SLOPE 12
#define GLOBAL_BUFF_SIZE 10000
#define LOCAL_BUFF_SIZE 10000

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
inline bool local_qWork(double c, double d, double * buffer, int * head, int * tail, int * status)
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

inline bool local_deqWork(double * c, double * d, double * buffer, int * head, int * tail, int * status)
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
inline bool local_setMax(double * currentMax, double fc, double fd)
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
inline bool validInterval(double currentMax, double c, double d)
{
	if(((f(c) + f(d) + SLOPE*(d - c))/2) > (currentMax + EPSILON))
		return true; 
	else
		return false;
}

// Returns the amount of the remaining interval represented in the buffer 
// as a percentage
// FOR DEBUGGING
double intervalLeft(double originalSize, double * buffer, int bufferSize, int head, int tail)
{
	double runSum = 0; 
	do
	{
		runSum += (buffer[head+1] - buffer[head]);
		head = (head+2)%bufferSize;
	}while(head != tail);
	
	return runSum/originalSize; 
}

// Returns the average size of the subintervals in the buffer
// FOR DEBUGGING ONLY
double averageSubintervalSize(double * buffer, int bufferSize, int head, int tail)
{
	double runSum = 0;
	int itemCount = 0;  
	do
	{
		runSum += (buffer[head+1] - buffer[head]);
		head = (head+2)%bufferSize;
		itemCount++; 
	}while(head != tail);
	
	return runSum/itemCount; 
}

// Prints the intervals in the buffer
// FOR DEBUGGING ONLY
void printBuff(double * buffer, int bufferSize, int head, int tail, int count)
{
	int iterCount = 0;  
	do
	{
		printf("[%f, %f]\t", buffer[head], buffer[head+1]);
		head = (head+2)%bufferSize;
		iterCount++; 
	}while(head != tail && iterCount < count);
	
	printf("\n");  
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

	int debugCount = 0; 

	do
	{
		// FOR DEBUGGING
		debugCount++; 
		if(debugCount == 1000)
		{
			printBuff(local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail, 10); 
			printf("Status: %d\tCapLeft: %d\tCurrentMax: %2.19f\tPercentLeft: %f\tAvgSubIntSize: %1.10f\n", local_status, local_tail%LOCAL_BUFF_SIZE - local_head%LOCAL_BUFF_SIZE, local_max, intervalLeft(END_B-START_A, local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail), averageSubintervalSize(local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail));
			debugCount = 0; 
		}
		
		//int wait = 0;
		//cin >> wait; 
		
		// Get work from queue
		local_deqWork(&local_c, &local_d, local_buffer, &local_head, &local_tail, &local_status);

		// Maybe reorganize to change max first? 
		// Check if possible larger
		if(validInterval(local_max, local_c, local_d))
		{
			// Maybe set max somewhere else for higher efficiency? 
			// Use the boolean output maybe? 
			local_setMax(&local_max, f(local_c), f(local_d)); 
			// Determine whether all of these are necessary 
			// Can only add subintervals if there is room for both
				// Else you have to add the original interval back and not the subintervals
			if((local_head%LOCAL_BUFF_SIZE) - (local_tail%LOCAL_BUFF_SIZE) == 2)
			{
				// Debugging
				//printf("Requeued\n"); 
				// Queue the original subinterval
				local_qWork(local_c, local_d, local_buffer, &local_head, &local_tail, &local_status); 
			}
			else
			{
				//if((local_d-local_c) > EPSILON)
					local_qWork(local_c, ((local_d-local_c)/2)+local_c, local_buffer, &local_head, &local_tail, &local_status);
			
				//if((local_d-local_c) > EPSILON)
					local_qWork(((local_d-local_c)/2)+local_c, local_d, local_buffer, &local_head, &local_tail, &local_status);	
			}
		}
	}while(local_status != STATUS_EMPTY); 

	return 0; 	 
}
