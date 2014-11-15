// Tentative questions: 
// In email, when describing f(x), the 3rd line is "z = x". Why isn't it z = 0? 
// Try setting number of threads to 1.5X number of procs to do hyperthreading? 
// ARE COMPILER OPTIMIZATIONS ALLOWED!?!?!? -O3? 

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
	

	// Global Stuff
	double global_max = 0; 
	double global_buffer[GLOBAL_BUFF_SIZE]; 
	int global_head = 0; 
	int global_tail = 0; 
	int global_status = STATUS_EMPTY; 
	bool * global_doneArray = new bool[numThreads]; 
	for(int i=0; i<numThreads; i++)
		global_doneArray[i] = false; 


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
				printBuff(local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail, 10); 
				printf("Status: %d\tSpaceLeft: %d\tCurMax: %2.8f\tPercentLeft: %f\tAvgSubIntSize: %1.8f\n", local_status, spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status), local_max, intervalLeft(END_B-START_A, local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail), averageSubintervalSize(local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail));
				debugCount = 0; 
			}
			
			int wait = 0;
			cin >> wait; 
			
			// Get work from a queue
			if(local_status != STATUS_EMPTY)
			{
				local_deqWork(&local_c, &local_d, local_buffer, &local_head, &local_tail, &local_status);
				global_doneArray[local_threadNum] = false; 
			}
			else
			{
				bool res; 
				global_doneArray[local_threadNum] = true; 
				while(!allDone(global_doneArray, numThreads))
				{
					res = global_deqWork(&local_c, &local_d, global_buffer, &global_head, &global_tail, &global_status);
					if(res)
						break;
				}
				if(!res)
					break; 
			}
				

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
				
				// IF FULL SEND WORK TO GLOBAL BUFF AT A RATE DETERMINED BY A CONSTANT
				if(spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status) == 2)
				{
					// Debugging
					//printf("Requeued\n"); 
					// Queue the original subinterval
					//printf("Interval Before Shrink: [%f, %f]\n", local_c, local_d);
					if(global_status == STATUS_FULL)
					{
						shrinkInterval(&local_max, &local_c, &local_d);
						local_qWork(local_c, local_d, local_buffer, &local_head, &local_tail, &local_status); 
					}
					else 
					{
						// Need function that will queue two items or none here
						//if(global_q2Work())
						//if(false)
						//else
						{
							shrinkInterval(&local_max, &local_c, &local_d);
							local_qWork(local_c, local_d, local_buffer, &local_head, &local_tail, &local_status); 
						}
							
					}
					//printf("Interval After Shrink:  [%f, %f]\n", local_c, local_d);
				}
				else
				{
					local_qWork(local_c, ((local_d-local_c)/2)+local_c, local_buffer, &local_head, &local_tail, &local_status);
					local_qWork(((local_d-local_c)/2)+local_c, local_d, local_buffer, &local_head, &local_tail, &local_status);	
				}
			}
		}while((local_status != STATUS_EMPTY) || (global_status != STATUS_EMPTY) || !allDone(global_doneArray, numThreads)); 
	} // END PARALLEL 

	printf("LocalMax = %2.30f\n", global_max); 
	return 0; 	 
}
