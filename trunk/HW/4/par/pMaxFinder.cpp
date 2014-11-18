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
	global_curMaxVal = 0;  
	global_front = 0; 
	global_back = 0; 
	global_buffState = STATUS_EMPTY; 

	// This array determines when ALL threads are finished
	global_dArray = new bool[numThreads]; 
	for(int i=0; i<numThreads; i++)
		global_dArray[i] = false;

	// For timing purposes
	double startTime = omp_get_wtime();	
	#pragma omp parallel 
	{
		// Init local variables
		double local_circalQueue[LOCAL_BUFF_SIZE]; 
		double local_c = 0;
		double local_d = 0; 
		int local_front = 0; 
		int local_back = 0; 
		int local_buffState = STATUS_EMPTY; 
	
		// Add first interval to queue
		int local_threadNum = omp_get_thread_num(); 
		local_qWork(local_threadNum*chunkSize+START_A, (local_threadNum+1)*chunkSize+START_A, local_circalQueue, &local_front, &local_back, &local_buffState);
		
		int debugCount = 0; 

		bool lContinue = true;
		while(lContinue)	
		{
			// For debugging
			printDiagOutput(&debugCount, local_front, local_back, local_buffState, local_threadNum, local_circalQueue);

			bool keepGoing = false;	
			// Get work from a queue
			if(local_buffState != STATUS_EMPTY)
			{
				// Local circalQueue still has work so we get some from there
				local_deqWork(&local_c, &local_d, local_circalQueue, &local_front, &local_back, &local_buffState);
				global_dArray[local_threadNum] = false; 
				keepGoing = true; 
			}
			
			else
			{
				global_dArray[local_threadNum] = true; 
				while(!allDone(global_dArray, numThreads) && !keepGoing)
				{
					if(global_buffState != STATUS_EMPTY)
					{
						keepGoing = global_safeWorkBuffer(FUN_DEQUEUE, &local_c, &local_d, 0, 0);
					}
				}
				if(keepGoing)
					global_dArray[local_threadNum] = false; 
				else
					global_dArray[local_threadNum] = true; 	
			}
		
			if(keepGoing)
			{	
				// Check if possible larger
				if(validInterval(global_curMaxVal, local_c, local_d))
				{
					global_setMax(mathFun(local_c), mathFun(local_d)); 
					
					// IF FULL, SEND WORK TO GLOBAL BUFF AT A RATE DETERMINED BY A CONSTANT

					// Two intervals will not fit in local circalQueue
					int local_locSpaceLeft = spaceLeft(LOCAL_BUFF_SIZE, local_front, local_back, local_buffState);
					int local_globSpaceLeft = spaceLeft(GLOBAL_BUFF_SIZE, global_front, global_back, global_buffState);
					//if(local_locSpaceLeft == 2 || !global_allWorking ||((local_globSpaceLeft > GLOBAL_BUFF_SIZE/10) && (local_locSpaceLeft < LOCAL_BUFF_SIZE/10)))
					//if(local_locSpaceLeft == 2 || ((local_globSpaceLeft > GLOBAL_BUFF_SIZE/2)))
					//if(spaceLeft(LOCAL_BUFF_SIZE, local_front, local_back, local_buffState) == 2 || spaceLeft(GLOBAL_BUFF_SIZE, global_front, global_back, global_buffState) > GLOBAL_BUFF_SIZE/2)
					if(spaceLeft(LOCAL_BUFF_SIZE, local_front, local_back, local_buffState) == 2)
					{
						// Global circalQueue is full too - so we shrink the current interval instead of splitting it
						if(global_buffState == STATUS_FULL)
						{
							// NEED TO FIX THIS FUNCTION BELOW
							shrinkInterval(global_curMaxVal, &local_c, &local_d);
							// Queue up shrunken interval back into local circalQueue
							local_qWork(local_c, local_d, local_circalQueue, &local_front, &local_back, &local_buffState); 
						}
						else 
						{
							double pC = local_c;
							double pD = ((local_d-local_c)/2)+local_c;
							double pC2 = ((local_d-local_c)/2)+local_c;
							double pD2 = local_d; 
							if(!global_safeWorkBuffer(FUN_DOUBLE_Q, &pC, &pD, pC2, pD2))
							{
								shrinkInterval(global_curMaxVal, &local_c, &local_d);
								local_qWork(local_c, local_d, local_circalQueue, &local_front, &local_back, &local_buffState); 
							}
								
						}
					}
					else
					{
						local_qWork(local_c, ((local_d-local_c)/2)+local_c, local_circalQueue, &local_front, &local_back, &local_buffState);
						local_qWork(((local_d-local_c)/2)+local_c, local_d, local_circalQueue, &local_front, &local_back, &local_buffState);	
					}
				}
	
				// Throws some to the global if necessary
				if(!global_allWorking && spaceLeft(LOCAL_BUFF_SIZE, local_front, local_back, local_buffState) < LOCAL_BUFF_SIZE/2)
				{
					for(int i=0; i<3; i++)
					{
						double tempC;
						double tempD; 
						if(local_deqWork(&tempC, &tempD, local_circalQueue, &local_front, &local_back, &local_buffState))
						{
							if(!global_safeWorkBuffer(FUN_SINGLE_Q, &tempC, &tempD, 0, 0))
							{
								shrinkInterval(global_curMaxVal, &local_c, &local_d);
								local_qWork(local_c, local_d, local_circalQueue, &local_front, &local_back, &local_buffState); 
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
		
	} // END PARALLEL 
	double endTime = omp_get_wtime(); 

	delete[] global_dArray;
	printf("GlobalMax = %2.30f in %f seconds\n\n", global_curMaxVal, endTime - startTime); 
	return 0; 	 
}
