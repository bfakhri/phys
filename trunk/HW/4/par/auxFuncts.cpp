#include "auxFuncts.h"


// Global Stuff
bool manager_allWorking; 
double manager_curMaxVal; 
double manager_circalQueue[GLOBAL_BUFF_SIZE];
int manager_front; 
int manager_back; 
int manager_buffState; 
bool * manager_dArray;

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

bool worker_qWork(double c, double d, double * circalQueue, int * front, int * back, int * curState)
{
	if(*curState == STATUS_FULL)
	{
		return false;
	}
	else
	{
		// Add to circular circalQueue
		circalQueue[*back] = c;
		circalQueue[*back+1] = d; 
		*back = (*back+2)%LOCAL_BUFF_SIZE;  
		if(*back == *front)
			*curState = STATUS_FULL;
		else
			*curState = STATUS_MID; 

		return true; 
	}
}

bool worker_deqWork(double * c, double * d, double * circalQueue, int * front, int * back, int * curState)
{
	if(*curState == STATUS_EMPTY)
	{
		return false;
	}
	else
	{
		// Get from circular circalQueue
		*c = circalQueue[*front];
		*d = circalQueue[*front+1]; 
		*front = (*front+2)%LOCAL_BUFF_SIZE;  
		if(*back == *front)
			*curState = STATUS_EMPTY;
		else
			*curState = STATUS_MID; 

		return true; 
	}
}


bool manager_safeWorkBuffer(int function, double * c, double * d, double c2, double d2)
{
	bool ret = true; 
	#pragma omp critical
	{
		// Dequeue function
		if(function == FUN_DEQUEUE)
		{
			if(manager_buffState == STATUS_EMPTY)
				ret = false;
			else
			{
				// Get from circular circalQueue
				*c = manager_circalQueue[manager_front];
				*d = manager_circalQueue[manager_front+1]; 
				manager_front = (manager_front+2)%GLOBAL_BUFF_SIZE;  
				if(manager_back == manager_front)
					manager_buffState = STATUS_EMPTY;
				else
					manager_buffState = STATUS_MID; 
			
			}

		}
		// Insert into circalQueue
		else
		{
			if(manager_buffState == STATUS_FULL)
			{
				ret = false;
			}
			else
			{
				// Check if inserting two intervals
				if(function == FUN_DOUBLE_Q)
				{
					if(spaceLeft(GLOBAL_BUFF_SIZE, manager_front, manager_back, manager_buffState) >= 4)
					{
						// Insert both intervals
						manager_circalQueue[manager_back] = *c;
						manager_circalQueue[manager_back+1] = *d; 
						manager_circalQueue[manager_back+2] = c2;
						manager_circalQueue[manager_back+3] = d2; 
						manager_back = (manager_back+4)%GLOBAL_BUFF_SIZE;  
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
					manager_circalQueue[manager_back] = *c;
					manager_circalQueue[manager_back+1] = *d; 
					manager_back = (manager_back+2)%GLOBAL_BUFF_SIZE;  
				}
				// Add to circular circalQueue
				if(manager_back == manager_front)
					manager_buffState = STATUS_FULL;
				else
					manager_buffState = STATUS_MID; 
			}
		}
	} // End Critical Section
	
	return ret; 
}

bool worker_peek(double * c, double * d, double * circalQueue, int * front, int * back, int * curState)
{
	if(*curState == STATUS_EMPTY)
	{
		return false;
	}
	else
	{
		// Add to circular circalQueue
		*c = circalQueue[*front];
		*d = circalQueue[*front+1]; 
		return true; 
	}
}

bool worker_setMax(double * currentMax, double fc, double fd)
{
	if(*currentMax + EPSILON < fc)
		*currentMax = fc;
	if(*currentMax + EPSILON < fd)
		*currentMax = fd; 
	else 
		return false; 

	return true; 
}

bool manager_setMax(double fc, double fd)
{
	bool ret = true; 
	#pragma omp critical
	{
		if(manager_curMaxVal + EPSILON < fc)
		{
			manager_curMaxVal = fc; 
		}
		else if(manager_curMaxVal + EPSILON < fd)
		{
			manager_curMaxVal = fd; 
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
	if(worker_setMax(currentMax, fC, fD))
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

int spaceLeft(int circalQueueSize, int front, int back, int curState)
{
	if(curState == STATUS_EMPTY)
		return circalQueueSize;
	else if(curState == STATUS_FULL)
		return 0; 
	else
	{
		if(back > front)
			return circalQueueSize - (back - front); 
		else
			return circalQueueSize - ((circalQueueSize - front) + back); 		
	}
}

// THIS VERSION ONLY WORKS WITH STACK VERSION OF q/deqWork
/*int spaceLeft(int circalQueueSize, int front, int back, int curState)
{
	if(curState == STATUS_EMPTY)
		return circalQueueSize;
	else if(curState == STATUS_FULL)
		return 0; 
	else
	{
		return (LOCAL_BUFF_SIZE-(front - 2));
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
		manager_allWorking = true; 
		return false;
	}
	// No threads are still working
	else if(doneCount >= size)
	{
		manager_allWorking = false; 
		return true; 
	}
	// Some but not all threads are still working
	else
	{
		manager_allWorking = false; 
		return false; 
	}
}

double intervalLeft(double originalSize, double * circalQueue, int circalQueueSize, int front, int back, int curState)
{
	if(curState == STATUS_EMPTY)
		return 0; 
	else 
	{
		double runSum = 0; 
		do
		{
			runSum += (circalQueue[front+1] - circalQueue[front]);
			front = (front+2)%circalQueueSize;
		}while(front != back);
		
		return 100*runSum/originalSize; 
	}
}

double averageSubintervalSize(double * circalQueue, int circalQueueSize, int front, int back, int curState)
{
	if(curState == STATUS_EMPTY)
		return 0; 
	else
	{
		double runSum = 0;
		int itemCount = 0;  
		do
		{
			runSum += (circalQueue[front+1] - circalQueue[front]);
			front = (front+2)%circalQueueSize;
			itemCount++; 
		}while(front != back);
		return runSum/(itemCount); 
	}
}

void printBuff(double * circalQueue, int circalQueueSize, int front, int back, int count)
{
	int iterCount = 0;  
	do
	{
		printf("[%f, %f]\t", circalQueue[front], circalQueue[front+1]);
		front = (front+2)%circalQueueSize;
		iterCount++; 
	}while(front != back && iterCount < count);
	
	printf("\n");  
}

void spinWait()
{
	while(1);
}

// For diagnostic output
void printDiagOutput(int * d, int worker_front, int worker_back, int worker_buffState, int worker_threadNum, double * worker_circalQueue)
{
	*d += 1; 
	if(*d == DEBUG_FREQ)
	{
		//printBuff(worker_circalQueue, LOCAL_BUFF_SIZE, worker_front, worker_back, 10); 
		printf("GlobalSpaceLeft: %d\t", spaceLeft(GLOBAL_BUFF_SIZE, manager_front, manager_back, manager_buffState));
		printf("tNum: %d\t\tStatus: %d\tSpacLeft: %d\t\tCurMax: %2.30f\tPercentLeft: %f\tAvgSubIntSize: %1.8f\n", worker_threadNum, worker_buffState, spaceLeft(LOCAL_BUFF_SIZE, worker_front, worker_back, worker_buffState), manager_curMaxVal, intervalLeft(END_B-START_A, worker_circalQueue, LOCAL_BUFF_SIZE, worker_front, worker_back, worker_buffState), averageSubintervalSize(worker_circalQueue, LOCAL_BUFF_SIZE, worker_front, worker_back, worker_buffState));
		*d = 0; 
	}
}
