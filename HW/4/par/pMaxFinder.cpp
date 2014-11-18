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

	// Initializing manager queue indicators
	manager_allWorking = true; 
	manager_curMaxVal = 0;  
	manager_front = 0; 
	manager_back = 0; 
	manager_buffState = STATUS_EMPTY; 

	// Init worker indicators 
	int worker_front = 0; 
	int worker_back = 0; 
	int worker_buffState = STATUS_EMPTY; 

	printf("\nUser Max Threads: %d\tProgram Max Threads: %d\tOMP Max Threads: %d\n", MAX_THREADS, numThreads, omp_get_max_threads()); 		

	manager_dArray = new bool[numThreads]; 
	for(int i=0; i<numThreads; i++)
		manager_dArray[i] = false;

	// For timing purposes
	double t0 = omp_get_wtime();	

	// Make each thread have an individual copy of those variables
	#pragma omp parallel firstprivate(worker_front, worker_back, worker_buffState) 
	{
		// Init worker resources
		double worker_circalQueue[WKR_BUFF_SIZE]; 
		double worker_c = 0;
		double worker_d = 0; 
	
		// Add first interval to queue
		int worker_threadNum = omp_get_thread_num(); 
		worker_qWork(worker_threadNum*chunkSize+START_A, (worker_threadNum+1)*chunkSize+START_A, worker_circalQueue, &worker_front, &worker_back, &worker_buffState);
		
		// For debugging
		int debugCount = 0; 
	
		// Repeats until done
		while(1)	
		{
			// For debugging
			//printDiagOutput(&debugCount, worker_front, worker_back, worker_buffState, worker_threadNum, worker_circalQueue);

			bool keepGoing = false;	
			// Get work from a queue
			if(worker_buffState != STATUS_EMPTY)
			{
				// Local circalQueue still has work so we get some from there
				worker_deqWork(&worker_c, &worker_d, worker_circalQueue, &worker_front, &worker_back, &worker_buffState);
				manager_dArray[worker_threadNum] = false; 
				keepGoing = true; 
			}			
			else
			{
				manager_dArray[worker_threadNum] = true; 
				while(!readyToLeave(manager_dArray, numThreads))
				{
					if(manager_buffState != STATUS_EMPTY)
					{
						keepGoing = manager_safeWorkBuffer(FUN_DEQUEUE, &worker_c, &worker_d, 0, 0);
						// Can continue because we now have work
						if(keepGoing)
							break;
					}
				}
				if(keepGoing)
					manager_dArray[worker_threadNum] = false; 
				else
					manager_dArray[worker_threadNum] = true; 	
			}
		
			if(!keepGoing)
			{
				// Done, leave loop
				break;	
			}
			else
			{
				// Check if possible larger
				if(promisingInterval(manager_curMaxVal, worker_c, worker_d))
				{
					manager_setMax(mathFun(worker_c), mathFun(worker_d)); 
					

					// Two intervals will not fit in worker queue
					int worker_locSpaceLeft = currentCapacity(WKR_BUFF_SIZE, worker_front, worker_back, worker_buffState);
					int worker_globSpaceLeft = currentCapacity(MGR_BUFF_SIZE, manager_front, manager_back, manager_buffState);
					//if(worker_locSpaceLeft == 2 || !manager_allWorking ||((worker_globSpaceLeft > MGR_BUFF_SIZE/10) && (worker_locSpaceLeft < WKR_BUFF_SIZE/10)))
					//if(worker_locSpaceLeft == 2 || ((worker_globSpaceLeft > MGR_BUFF_SIZE/2)))
					//if(currentCapacity(WKR_BUFF_SIZE, worker_front, worker_back, worker_buffState) == 2 || currentCapacity(MGR_BUFF_SIZE, manager_front, manager_back, manager_buffState) > MGR_BUFF_SIZE/2)
					if(currentCapacity(WKR_BUFF_SIZE, worker_front, worker_back, worker_buffState) == 2)
					{
						// if manger queue is full too - so we cut down the current interval instead of splitting it
						if(manager_buffState == STATUS_FULL)
						{
							cutInterval(manager_curMaxVal, &worker_c, &worker_d);
							// Queue up shrunken interval back into worker queue
							worker_qWork(worker_c, worker_d, worker_circalQueue, &worker_front, &worker_back, &worker_buffState); 
						}
						else 
						{
							double pC = worker_c, pD = ((worker_d-worker_c)/2)+worker_c, pC2 = ((worker_d-worker_c)/2)+worker_c, pD2 = worker_d; 
							// If all else fails and we have NO ROOM 
							// we must add this interval back to the buffer
							// but to make progress we shrink it
							if(!manager_safeWorkBuffer(FUN_DOUBLE_Q, &pC, &pD, pC2, pD2))
							{
								cutInterval(manager_curMaxVal, &worker_c, &worker_d);
								worker_qWork(worker_c, worker_d, worker_circalQueue, &worker_front, &worker_back, &worker_buffState); 
							}
								
						}
					}
					else
					{
						worker_qWork(worker_c, ((worker_d-worker_c)/2)+worker_c, worker_circalQueue, &worker_front, &worker_back, &worker_buffState);
						worker_qWork(((worker_d-worker_c)/2)+worker_c, worker_d, worker_circalQueue, &worker_front, &worker_back, &worker_buffState);	
					}
				}
	
				// Throws some to the manager if necessary
				if(!manager_allWorking && currentCapacity(WKR_BUFF_SIZE, worker_front, worker_back, worker_buffState) < WKR_BUFF_SIZE/2)
				{
					for(int i=0; i<3; i++)
					{
						double tempC;
						double tempD; 
						if(worker_deqWork(&tempC, &tempD, worker_circalQueue, &worker_front, &worker_back, &worker_buffState))
						{
							if(!manager_safeWorkBuffer(FUN_SINGLE_Q, &tempC, &tempD, 0, 0))
							{
								cutInterval(manager_curMaxVal, &worker_c, &worker_d);
								worker_qWork(worker_c, worker_d, worker_circalQueue, &worker_front, &worker_back, &worker_buffState); 
							}
						}
					}
				}
			}
		}
		
	} // END PARALLEL 
	double t1 = omp_get_wtime(); 

	delete[] manager_dArray;
	printf("GlobalMax = %2.30f in %f seconds\n\n", manager_curMaxVal, t1 - t0); 
	return 0; 	 
}
