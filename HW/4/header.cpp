#include "header.h"

// Function we want to find the maximum of
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

// DEBUGGING
/*
double f(double x)
{
	return 1;
}
*/

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

// Gives front value but does not pop it off the queue
bool local_peek(double * c, double * d, double * buffer, int * head, int * tail, int * status)
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

bool validInterval(double currentMax, double c, double d)
{
	if(((f(c) + f(d) + SLOPE*(d - c))/2) > (currentMax + EPSILON))
		return true; 
	else
		return false;
}

// FOR DEBUGGING ONLY
/*
inline bool validInterval(double cu, double c, double d)
{
	return true; 
}
*/

// Attempts to rid itself of a piece of the interval handed to it
bool shrinkInterval(double currentMax, double * c, double * d)
{
	// Save the original values
	double C = *c; 
	double D = *d; 
	
	// Shrink from the left side
	while(validInterval(currentMax, C, D))
	{
		//printf("stuck"); 
		D = (D - C)/2 + C; 
	}

	//printf("\nNOT STUCK\n"); 	
	*c = D;
	C = D; 
	D = *d; 	

	// Shrink from the right side
	while(validInterval(currentMax, C, D))
	{
		C = (D - C)/2 + C; 
	}

	*d = C; 
	//*c = retC; 
	
	//printf("Getting Out"); 
	// THIS SHOULD CHECK IF FAILED OR NOT, SOMEHOW? 
	return true; 
}

// Returns space left in buffer 
int spaceLeft(int bufferSize, int head, int tail, int status)
{
	if(status == STATUS_EMPTY)
		return bufferSize;
	else if(status == STATUS_FULL)
		return 0; 
	else
	{
		if(tail > head)
			return bufferSize - (tail - head); 
		else
			return bufferSize - ((bufferSize - head) + tail); 		
	}
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
	
	return 100*runSum/originalSize; 
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
