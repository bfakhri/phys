// Tentative questions: 
// In email, when describing f(x), the 3rd line is "z = x". Why isn't it z = 0? 
// Try setting number of threads to 1.5X number of procs to do hyperthreading? 
// ARE COMPILER OPTIMIZATIONS ALLOWED!?!?!? -O3? 

#include "header.h"

using namespace std; 


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
			printf("Status: %d\tSpaceLeft: %d\tCurMax: %2.8f\tPercentLeft: %f\tAvgSubIntSize: %1.8f\n", local_status, spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status), local_max, intervalLeft(END_B-START_A, local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail), averageSubintervalSize(local_buffer, LOCAL_BUFF_SIZE, local_head, local_tail));
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
			if(spaceLeft(LOCAL_BUFF_SIZE, local_head, local_tail, local_status) == 2)
			{
				// Debugging
				//printf("Requeued\n"); 
				// Queue the original subinterval
				//printf("Interval Before Shrink: [%f, %f]\n", local_c, local_d);
				shrinkInterval(local_max, &local_c, &local_d);
				//printf("Interval After Shrink:  [%f, %f]\n", local_c, local_d);
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

	printf("LocalMax = %2.30f\n", local_max); 
	return 0; 	 
}
