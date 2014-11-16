// Tentative questions: 
// In email, when describing f(x), the 3rd line is "z = x". Why isn't it z = 0? 
// Try setting number of threads to 1.5X number of procs to do hyperthreading? 
// ARE COMPILER OPTIMIZATIONS ALLOWED!?!?!? -O3? 
// Change order by which elements are added to global buffer 
// 	try to take elements from the from of queue rather than those that would have been 
//	added to the back. 

#include "header.h"

using namespace std; 


int main()
{
	// Set number of threads
	// TRY HYPERTHREADING? 
	int numThreads = omp_get_num_procs(); 	
	omp_set_num_threads(numThreads);

	// Vars to be used  
	double intervalSpan = END_B - START_A;
	double chunkSize = intervalSpan/numThreads;

	printf("Number of threads: %d\n\n", numThreads); 	

	global_initBuffer(); 
	bool * global_doneArray = new bool[numThreads]; 
	for(int i=0; i<numThreads; i++)
		global_doneArray[i] = false;

	// FOR DEBUGGING
	/*double cccc = 1; 
	double dddd = 10;
	global_safeWorkBuffer(FUN_SINGLE_Q,  &cccc, &dddd, 0, 0); 
	for(int i=0; i<numThreads; i++)
	{
		cccc = i*chunkSize + START_A;
		dddd = (i+1)*chunkSize + START_A; 
		global_safeWorkBuffer(FUN_SINGLE_Q,  &cccc, &dddd, 0, 0); 
	}
	*/
	#pragma omp parallel 
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
		int local_threadNum = omp_get_thread_num(); 
		local_qWork(local_threadNum*chunkSize+START_A, (local_threadNum+1)*chunkSize+START_A, local_buffer, &local_head, &local_tail, &local_status);

		int debugCount = 0; 

		do
		{
			// FOR DEBUGGING
			debugCount++; 
			if(debugCount == DEBUG_FREQ)
			{
				//printBuff(local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail, 10); 
				printf("Global Buff Space Left: %d\n", spaceLeft(GLOBAL_BUFF_SIZE, global_head, global_tail, global_status));
				printf("tNum: %d\tStatus: %d\tSpaceLeft: %d\tCurMax: %2.30f\tPercentLeft: %f\tAvgSubIntSize: %1.8f\n", local_threadNum, local_status, spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status), global_max, intervalLeft(END_B-START_A, local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail), averageSubintervalSize(local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail));
				debugCount = 0; 
			}
			
			bool cont = false;	
			// Get work from a queue
			if(local_status != STATUS_EMPTY)
			{
				// Local buffer still has work so we get some from there
				local_deqWork(&local_c, &local_d, local_buffer, &local_head, &local_tail, &local_status);
				global_doneArray[local_threadNum] = false; 
				cont = true; 
			}
			else
			{
				// Need work so request some from global buffer 
				global_doneArray[local_threadNum] = true; 
				while(!allDone(global_doneArray, numThreads) && !cont)
				{
					cont = global_safeWorkBuffer(FUN_DEQUEUE, &local_c, &local_d, 0, 0);
					//if(!cont)
					//	sleep(1); 
				}
				if(cont)
					global_doneArray[local_threadNum] = false; 
			}

			if(cont)
			{	
				// Check if possible larger
				if(validInterval(global_max, local_c, local_d))
				{
					global_setMax(f(local_c), f(local_d)); 
					
					// IF FULL, SEND WORK TO GLOBAL BUFF AT A RATE DETERMINED BY A CONSTANT

					// Two intervals will not fit in local buffer
					if(spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status) == 2)
					{
						// Global buffer is full too - so we shrink the current interval instead of splitting it
						if(global_status == STATUS_FULL)
						{
							// NEED TO FIX THIS FUNCTION BELOW
							shrinkInterval(global_max, &local_c, &local_d);
							// Queue up shrunken interval back into local buffer
							local_qWork(local_c, local_d, local_buffer, &local_head, &local_tail, &local_status); 
						}
						else 
						{
							double pC = local_c;
							double pD = ((local_d-local_c)/2)+local_c;
							double pC2 = ((local_d-local_c)/2)+local_c;
							double pD2 = local_d; 
							if(!global_safeWorkBuffer(FUN_DOUBLE_Q, &pC, &pD, pC2, pD2))
							{
								shrinkInterval(global_max, &local_c, &local_d);
								local_qWork(local_c, local_d, local_buffer, &local_head, &local_tail, &local_status); 
							}
								
						}
					}
					else
					{
						local_qWork(local_c, ((local_d-local_c)/2)+local_c, local_buffer, &local_head, &local_tail, &local_status);
						local_qWork(((local_d-local_c)/2)+local_c, local_d, local_buffer, &local_head, &local_tail, &local_status);	
					}
				}
			}
			else
			{
				break;
			}
		}while((local_status != STATUS_EMPTY) || !allDone(global_doneArray, numThreads)); 
		
		printf("Thread %d is shutting down\n", local_threadNum);
		global_doneArray[local_threadNum] = true;
		for(int i=0; i<numThreads; i++)
			printf("\tThread %d has status: %s", i, global_doneArray[i] ? "true" : "false"); 
		printf("\nGlobalMax = %2.30f\n", global_max); 
	} // END PARALLEL 

	printf("GlobalMax = %2.30f\n", global_max); 
	return 0; 	 
}