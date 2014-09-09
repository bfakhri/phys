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
	for(int step=0; !done; step++)
	{
		// Send - if true this proc is embassador
		if((procID+1)%(pow(2,step)) == 0)
		{
			// Add check for end right here
			for(int i=1; i<=(pow(2,step)); i++)
				send(procID+i, myValue); 
		}
		
		// Receive 
		if(step == 0 && myID != 0)
		{
			receive(myID-1, inComingValue); 
		}
		else
		{
				if
		}
					

		// Compare
		if(inComingValue > myValue)
			myValue = inComingValue; 
	}


