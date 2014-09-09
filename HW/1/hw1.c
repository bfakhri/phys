// Name: Bijan Fakhri
// Date: 09/09/2014
// Assignment: 1

// This document is the programming portion of the assignment.
// The O-notational analysis is not included here. 


// Problem 1 - Fully Connected Computer
// The program calculates scan(max) on the 
// array of processors that are fully connected.

int scanMax()
{
	int myValue = getProcessorValue(); 
	// Holds the total number of processors (p)
	int maxProcs = getMaxProcessors(); 
	// Holds the processor ID for this processor
	int procID = getProcID(); 

	int step = 0; 
	for(int step=0; pow(2, step) < maxProcs; step++)
	{
		group = pow(2, step); 
		// Send - if true this proc is embassador
		if((procID+1)%group) == 0)
		{
			// Add check for end right here
			for(int i=1; i<=group; i++)
				send(procID+i, myValue); 
		}
		
		// Receive 
		if(myID != 0)
		{
			if(step == 0)
			{
				receive(myID-1, inComingValue); 
			}
			else
			{
				if((myID%(group*2)) >= group)
				{
					receive((myID/group)*group-1, inComingValue);
				}
			}
		}
					
		// Compare
		if(inComingValue > myValue)
			myValue = inComingValue; 
	}
	return inComingValue; 
}


