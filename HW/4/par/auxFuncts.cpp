#include "auxFuncts.h"


// Global Stuff
bool global_allWorking; 
double global_max; 
double global_buffer[GLOBAL_BUFF_SIZE];
int global_head; 
int global_tail; 
int global_status; 

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
		// Get from circular buffer
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


bool global_safeWorkBuffer(int function, double * c, double * d, double c2, double d2)
{
	bool ret = true; 
	#pragma omp critical
	{
		// Dequeue function
		if(function == FUN_DEQUEUE)
		{
			if(global_status == STATUS_EMPTY)
				ret = false;
			else
			{
				// Get from circular buffer
				*c = global_buffer[global_head];
				*d = global_buffer[global_head+1]; 
				global_head = (global_head+2)%GLOBAL_BUFF_SIZE;  
				if(global_tail == global_head)
					global_status = STATUS_EMPTY;
				else
					global_status = STATUS_MID; 
			
			}

		}
		// Insert into buffer
		else
		{
			if(global_status == STATUS_FULL)
			{
				ret = false;
			}
			else
			{
				// Check if inserting two intervals
				if(function == FUN_DOUBLE_Q)
				{
					if(spaceLeft(GLOBAL_BUFF_SIZE, global_head, global_tail, global_status) >= 4)
					{
						// Insert both intervals
						global_buffer[global_tail] = *c;
						global_buffer[global_tail+1] = *d; 
						global_buffer[global_tail+2] = c2;
						global_buffer[global_tail+3] = d2; 
						global_tail = (global_tail+4)%GLOBAL_BUFF_SIZE;  
					}
					else
					{
						// Cannot insert both succesfully so NONE will be inserted
						ret = false; 
					}
				}
				else
				{
					// Already checked to make sure it is not full so insert
					global_buffer[global_tail] = *c;
					global_buffer[global_tail+1] = *d; 
					global_tail = (global_tail+2)%GLOBAL_BUFF_SIZE;  
				}
				// Add to circular buffer
				if(global_tail == global_head)
					global_status = STATUS_FULL;
				else
					global_status = STATUS_MID; 
			}
		}
	} // End Critical Section
	
	return ret; 
}

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

bool local_setMax(double * currentMax, double fc, double fd)
{
	if(*currentMax + EPSILON < fc)
		*currentMax = fc;
	if(*currentMax + EPSILON < fd)
		*currentMax = fd; 
	else 
		return false; 

	return true; 
}

bool global_setMax(double fc, double fd)
{
	bool ret = true; 
	#pragma omp critical
	{
		if(global_max + EPSILON < fc)
		{
			global_max = fc; 
		}
		else if(global_max + EPSILON < fd)
		{
			global_max = fd; 
		}
		else
		{ 
			ret = false; 
		}
	}
	return ret; 
}

bool validInterval(double currentMax, double c, double d)
{
	if((SLOPE*(d - c)) < EPSILON)
		return false; 
	if(((f(c) + f(d) + SLOPE*(d - c))/2) > (currentMax + EPSILON))
		return true; 
	else
		return false;
}

bool validIntervalAndMax(double * currentMax, double c, double d)
{
	double fC = f(c); 
	double fD = f(d); 
	if(local_setMax(currentMax, fC, fD))
	{
		return true; 
	}
	else
	{
		if(((fC + fD + SLOPE*(d - c))/2) > (*currentMax + EPSILON))
			return true; 
		else
			return false;
	}
}



bool shrinkInterval(double currentMax, double * c, double * d)
{
	// Save the original values
	double C = *c; 
	double D = *d; 
	
	// Shrink from the left side
	while(validInterval(currentMax, C, D))
	{
		D = (D - C)/2 + C; 
	}

	*c = D;
	C = D; 
	D = *d; 	

	// Shrink from the right side
	while(validInterval(currentMax, C, D))
	{
		C = (D - C)/2 + C; 
	}

	*d = C; 
	
	return true; 
}

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

// THIS VERSION ONLY WORKS WITH STACK VERSION OF q/deqWork
/*int spaceLeft(int bufferSize, int head, int tail, int status)
{
	if(status == STATUS_EMPTY)
		return bufferSize;
	else if(status == STATUS_FULL)
		return 0; 
	else
	{
		return (LOCAL_BUFF_SIZE-(head - 2));
	}
}*/

bool allDone(bool * doneArr, int size)
{
	// Count how many threads are done
	int doneCount = 0; 
	for(int i=0; i<size; i++)
	{
		if(doneArr[i])
			doneCount++; 
	}
	
	// All threads are still working
	if(doneCount == 0)
	{
		global_allWorking = true; 
		return false;
	}
	// No threads are still working
	else if(doneCount >= size)
	{
		global_allWorking = false; 
		return true; 
	}
	// Some but not all threads are still working
	else
	{
		global_allWorking = false; 
		return false; 
	}
}

double intervalLeft(double originalSize, double * buffer, int bufferSize, int head, int tail, int status)
{
	if(status == STATUS_EMPTY)
		return 0; 
	else 
	{
		double runSum = 0; 
		do
		{
			runSum += (buffer[head+1] - buffer[head]);
			head = (head+2)%bufferSize;
		}while(head != tail);
		
		return 100*runSum/originalSize; 
	}
}

double averageSubintervalSize(double * buffer, int bufferSize, int head, int tail, int status)
{
	if(status == STATUS_EMPTY)
		return 0; 
	else
	{
		double runSum = 0;
		int itemCount = 0;  
		do
		{
			runSum += (buffer[head+1] - buffer[head]);
			head = (head+2)%bufferSize;
			itemCount++; 
		}while(head != tail);
		return runSum/(itemCount); 
	}
}

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

void spinWait()
{
	while(1);
}

// For diagnostic output
void printDiagOutput(int * d, int local_head, int local_tail, int local_status, int local_threadNum, double * local_buffer);
