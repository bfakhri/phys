#include "auxFuncts.h"


// Global Stuff
bool global_allWorking; 
double global_curMaxVal; 
double global_circalQueue[GLOBAL_BUFF_SIZE];
int global_front; 
int global_back; 
int global_buffState; 
bool * global_dArray;

double mathFun(double x)
{
	double outside = 0; 
	for(unsigned int i = 100; i >= 1; --i)
	{
		double inside = 0; 
		for(unsigned int j = i; j >= 1; --j)
		{
			inside += pow((x + j), -3.1f);
		}
		
		outside += sin(x + inside)/pow(1.2f, i);
	}

	return outside; 
}

bool local_qWork(double c, double d, double * circalQueue, int * head, int * tail, int * status)
{
	if(*status == STATUS_FULL)
	{
		return false;
	}
	else
	{
		// Add to circular circalQueue
		circalQueue[*tail] = c;
		circalQueue[*tail+1] = d; 
		*tail = (*tail+2)%LOCAL_BUFF_SIZE;  
		if(*tail == *head)
			*status = STATUS_FULL;
		else
			*status = STATUS_MID; 

		return true; 
	}
}

bool local_deqWork(double * c, double * d, double * circalQueue, int * head, int * tail, int * status)
{
	if(*status == STATUS_EMPTY)
	{
		return false;
	}
	else
	{
		// Get from circular circalQueue
		*c = circalQueue[*head];
		*d = circalQueue[*head+1]; 
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
			if(global_buffState == STATUS_EMPTY)
				ret = false;
			else
			{
				// Get from circular circalQueue
				*c = global_circalQueue[global_front];
				*d = global_circalQueue[global_front+1]; 
				global_front = (global_front+2)%GLOBAL_BUFF_SIZE;  
				if(global_back == global_front)
					global_buffState = STATUS_EMPTY;
				else
					global_buffState = STATUS_MID; 
			
			}

		}
		// Insert into circalQueue
		else
		{
			if(global_buffState == STATUS_FULL)
			{
				ret = false;
			}
			else
			{
				// Check if inserting two intervals
				if(function == FUN_DOUBLE_Q)
				{
					if(spaceLeft(GLOBAL_BUFF_SIZE, global_front, global_back, global_buffState) >= 4)
					{
						// Insert both intervals
						global_circalQueue[global_back] = *c;
						global_circalQueue[global_back+1] = *d; 
						global_circalQueue[global_back+2] = c2;
						global_circalQueue[global_back+3] = d2; 
						global_back = (global_back+4)%GLOBAL_BUFF_SIZE;  
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
					global_circalQueue[global_back] = *c;
					global_circalQueue[global_back+1] = *d; 
					global_back = (global_back+2)%GLOBAL_BUFF_SIZE;  
				}
				// Add to circular circalQueue
				if(global_back == global_front)
					global_buffState = STATUS_FULL;
				else
					global_buffState = STATUS_MID; 
			}
		}
	} // End Critical Section
	
	return ret; 
}

bool local_peek(double * c, double * d, double * circalQueue, int * head, int * tail, int * status)
{
	if(*status == STATUS_EMPTY)
	{
		return false;
	}
	else
	{
		// Add to circular circalQueue
		*c = circalQueue[*head];
		*d = circalQueue[*head+1]; 
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
		if(global_curMaxVal + EPSILON < fc)
		{
			global_curMaxVal = fc; 
		}
		else if(global_curMaxVal + EPSILON < fd)
		{
			global_curMaxVal = fd; 
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
	if(((mathFun(c) + mathFun(d) + SLOPE*(d - c))/2) > (currentMax + EPSILON))
		return true; 
	else
		return false;
}

bool validIntervalAndMax(double * currentMax, double c, double d)
{
	double fC = mathFun(c); 
	double fD = mathFun(d); 
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

int spaceLeft(int circalQueueSize, int head, int tail, int status)
{
	if(status == STATUS_EMPTY)
		return circalQueueSize;
	else if(status == STATUS_FULL)
		return 0; 
	else
	{
		if(tail > head)
			return circalQueueSize - (tail - head); 
		else
			return circalQueueSize - ((circalQueueSize - head) + tail); 		
	}
}

// THIS VERSION ONLY WORKS WITH STACK VERSION OF q/deqWork
/*int spaceLeft(int circalQueueSize, int head, int tail, int status)
{
	if(status == STATUS_EMPTY)
		return circalQueueSize;
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

double intervalLeft(double originalSize, double * circalQueue, int circalQueueSize, int head, int tail, int status)
{
	if(status == STATUS_EMPTY)
		return 0; 
	else 
	{
		double runSum = 0; 
		do
		{
			runSum += (circalQueue[head+1] - circalQueue[head]);
			head = (head+2)%circalQueueSize;
		}while(head != tail);
		
		return 100*runSum/originalSize; 
	}
}

double averageSubintervalSize(double * circalQueue, int circalQueueSize, int head, int tail, int status)
{
	if(status == STATUS_EMPTY)
		return 0; 
	else
	{
		double runSum = 0;
		int itemCount = 0;  
		do
		{
			runSum += (circalQueue[head+1] - circalQueue[head]);
			head = (head+2)%circalQueueSize;
			itemCount++; 
		}while(head != tail);
		return runSum/(itemCount); 
	}
}

void printBuff(double * circalQueue, int circalQueueSize, int head, int tail, int count)
{
	int iterCount = 0;  
	do
	{
		printf("[%f, %f]\t", circalQueue[head], circalQueue[head+1]);
		head = (head+2)%circalQueueSize;
		iterCount++; 
	}while(head != tail && iterCount < count);
	
	printf("\n");  
}

void spinWait()
{
	while(1);
}

// For diagnostic output
void printDiagOutput(int * d, int local_front, int local_back, int local_buffState, int local_threadNum, double * local_circalQueue)
{
	*d += 1; 
	if(*d == DEBUG_FREQ)
	{
		//printBuff(local_circalQueue, LOCAL_BUFF_SIZE, local_front, local_back, 10); 
		printf("GlobalSpaceLeft: %d\t", spaceLeft(GLOBAL_BUFF_SIZE, global_front, global_back, global_buffState));
		printf("tNum: %d\t\tStatus: %d\tSpacLeft: %d\t\tCurMax: %2.30f\tPercentLeft: %f\tAvgSubIntSize: %1.8f\n", local_threadNum, local_buffState, spaceLeft(LOCAL_BUFF_SIZE, local_front, local_back, local_buffState), global_curMaxVal, intervalLeft(END_B-START_A, local_circalQueue, LOCAL_BUFF_SIZE, local_front, local_back, local_buffState), averageSubintervalSize(local_circalQueue, LOCAL_BUFF_SIZE, local_front, local_back, local_buffState));
		*d = 0; 
	}
}
