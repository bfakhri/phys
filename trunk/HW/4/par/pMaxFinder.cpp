// Tentative questions: 
// In email, when describing f(x), the 3rd line is "z = x". Why isn't it z = 0? 
// Try setting number of threads to 1.5X number of procs to do hyperthreading? 
// ARE COMPILER OPTIMIZATIONS ALLOWED!?!?!? -O3? 
// Change order by which elements are added to global buffer 
// 	try to take elements from the from of queue rather than those that would have been 
//	added to the back. 

#include "auxFuncts.h"
	
using namespace std; 

int main(int argc, char * argv[])
{ 
	int MAX_THREADS = 0; 
	if(argc > 1)
	{
		MAX_THREADS = atoi(argv[1]); 		
	}
	// Set number of threads
	// Default is the number of processors
	int numThreads;
	if(MAX_THREADS == 0) 
		numThreads = omp_get_num_procs(); 	
	else
		numThreads = MAX_THREADS; 

	omp_set_num_threads(numThreads);

	// Vars to be used  
	double intervalSpan = END_B - START_A;
	double chunkSize = intervalSpan/numThreads;

	printf("\nUser Max Threads: %d\tProgram Max Threads: %d\tOMP Max Threads: %d\n", MAX_THREADS, numThreads, omp_get_max_threads()); 	
	
	// Initializing global indicators
	global_allWorking = true; 
	global_max = 0;  
	global_head = 0; 
	global_tail = 0; 
	global_status = STATUS_EMPTY; 

	// This array determines when ALL threads are finished
	bool * global_doneArray = new bool[numThreads]; 
	for(int i=0; i<numThreads; i++)
		global_doneArray[i] = false;

	// For timing purposes
	double startTime = omp_get_wtime();	
	#pragma omp parallel 
	{
		// Init local variables
		double local_buffer[LOCAL_BUFF_SIZE]; 
		double local_c = 0;
		double local_d = 0; 
		int local_head = 0; 
		int local_tail = 0; 
		int local_status = STATUS_EMPTY; 
	
		// Add first interval to queue
		int local_threadNum = omp_get_thread_num(); 
		local_qWork(local_threadNum*chunkSize+START_A, (local_threadNum+1)*chunkSize+START_A, local_buffer, &local_head, &local_tail, &local_status);
		
		int debugCount = 0; 

		bool lContinue = true;
		while(lContinue)	
		{
			// For debugging
			printDiagOutput(&debugCount, local_head, local_tail, local_status, local_threadNum, local_buffer);

			bool keepGoing = false;	
			// Get work from a queue
			if(local_status != STATUS_EMPTY)
			{
				// Local buffer still has work so we get some from there
				local_deqWork(&local_c, &local_d, local_buffer, &local_head, &local_tail, &local_status);
				global_doneArray[local_threadNum] = false; 
				keepGoing = true; 
			}
			
			else
			{
				global_doneArray[local_threadNum] = true; 
				while(!allDone(global_doneArray, numThreads) && !keepGoing)
				{
					if(global_status != STATUS_EMPTY)
					{
						keepGoing = global_safeWorkBuffer(FUN_DEQUEUE, &local_c, &local_d, 0, 0);
					}
				}
				if(keepGoing)
				{
					global_doneArray[local_threadNum] = false; 
				}
			}
		
			if(keepGoing)
			{	
				// Check if possible larger
				if(validInterval(global_max, local_c, local_d))
				{
					global_setMax(mathFun(local_c), mathFun(local_d)); 
					
					// IF FULL, SEND WORK TO GLOBAL BUFF AT A RATE DETERMINED BY A CONSTANT

					// Two intervals will not fit in local buffer
					int local_locSpaceLeft = spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status);
					int local_globSpaceLeft = spaceLeft(GLOBAL_BUFF_SIZE, global_head, global_tail, global_status);
					//if(local_locSpaceLeft == 2 || !global_allWorking ||((local_globSpaceLeft > GLOBAL_BUFF_SIZE/10) && (local_locSpaceLeft < LOCAL_BUFF_SIZE/10)))
					//if(local_locSpaceLeft == 2 || ((local_globSpaceLeft > GLOBAL_BUFF_SIZE/2)))
					//if(spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status) == 2 || spaceLeft(GLOBAL_BUFF_SIZE, global_head, global_tail, global_status) > GLOBAL_BUFF_SIZE/2)
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
	
				// Throws some to the global if necessary
				if(!global_allWorking && spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status) < LOCAL_BUFF_SIZE/2)
				{
					for(int i=0; i<3; i++)
					{
						double tempC;
						double tempD; 
						if(local_deqWork(&tempC, &tempD, local_buffer, &local_head, &local_tail, &local_status))
						{
							if(!global_safeWorkBuffer(FUN_SINGLE_Q, &tempC, &tempD, 0, 0))
							{
								shrinkInterval(global_max, &local_c, &local_d);
								local_qWork(local_c, local_d, local_buffer, &local_head, &local_tail, &local_status); 
							}
						}
					}
				}
			}
			else
			{
				lContinue = false; 
			}
		}
		
		global_doneArray[local_threadNum] = true; 	
	} // END PARALLEL 
	double endTime = omp_get_wtime(); 

	delete[] global_doneArray;
	printf("GlobalMax = %2.30f in %f seconds\n\n", global_max, endTime - startTime); 
	return 0; 	 
}
