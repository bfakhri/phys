#include <stdio.h> 
#include <stdlib.h>
#include <math.h> 
#include <mpi.h>
#include <iostream> 

int main(int argc, char *argv[])
{
   	int nodeRank, numNodes, length; 
	char name[128];  
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&nodeRank);
	MPI_Comm_size(MPI_COMM_WORLD,&numNodes);
	MPI_Get_processor_name(name, &length); 

	printf("nodeRank: %d\tnumNodes: %d\tProcName: %s\n", nodeRank, numNodes, name); 
	
	if(nodeRank == 0)
	{
		if(argc != 3)
		{
			std::cout << "Not enough or too many arguments\nExpecting 2" << std::endl; 
			return 1; 
		}
	}

	unsigned int N, timeSteps;
    	N = atoi(argv[1]); 
	timeSteps = atoi(argv[2]);

	// Find section sizes	
	unsigned int rowsPerSection = (N)/numNodes;
	unsigned int remainder = (N)%numNodes;
	if((nodeRank+1) <= remainder)
		rowsPerSection++;
	


	// Array to hold the values of cells for nodal subdivision
	double ** localMatrix = malloc(sizeof(double*)*rowsPerSection);
	for(unsigned int i=0; i<rowsPerSection; i++)
	{
		localMatrix[i] = malloc(sizeof(double)*N); 
	}
	
	// Fill in the matrix with initial values
	for(unsigned int r=0; r<rowsPerSection; r++)
	{
		for(unsigned int c=0; c<N; c++)
		{
			localMatrix[r][c] = 0.5f;
		}
	}
	if(nodeRank == 0)
	{
		for(unsigned int c=0; c<N; c++)
		{
			localMatrix[0][c] = 0;
		}
	}
	if(nodeRank == (numNodes-1))
	{
		for(unsigned int c=0; c<N; c++)
		{
			localMatrix[0][c] = SPECIALFUNCTIONOMGOMGOMG();
		}
	}

	// Take out unnecesary ones for top and bottom sections	
	double * lowerGhostRow = (double*)malloc(sizeof(double)*N);
	double * upperGhostRow = (double*)malloc(sizeof(double)*N);

	MPI_Request * sendDownReq = new MPI_Request;
	MPI_Request * sendUpReq = new MPI_Request;

	for(unsigned int t=0; t<timeSteps; t++)
	{
	
		// Send
		if(numNodes > 1)
		{
			if(nodeRank == 0)
			{
				// Receive from lower - sets up buffer
				MPI_Irecv((void*)lowerGhostRow, N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
				// Send down
				MPI_Isend((void*)localMatrix[rowsPerSection-1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
			}
			else if(nodeRank == (numNodes-1))
			{
				// Receive from upper - sets up buffer
				MPI_Irecv((void*)upperGhostRow, N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendUpReq);
				// Send up
				MPI_Isend((void*)localMatrix[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendUpReq);
			}else
			{
				
				// Receive from lower - sets up buffer
				MPI_Irecv((void*)lowerGhostRow, N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
				// Receive from upper - sets up buffer
				MPI_Irecv((void*)upperGhostRow, N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendUpReq);
				// Send down
				MPI_Isend((void*)localMatrix[rowsPerSection-1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
				// Send up
				MPI_Isend((void*)localMatrix[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendUpReq);
			}
		}

		// Compute nodes independent of ghost rows
		for(unsigned int row = 0; row < rowsPerSection; row++)
		{
			for(unsigned int column = 0; column < N; column++)
			{

				

	// Compute
	// Receive
	// compute
	
	}


	// Reduce to get global max value in globMax 
	MPI_Allreduce((void*)&nodeMax, (void*)&globMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
	
	for(int t=0; t<timeSteps; t++)
	{
	}

	MPI_Finalize();    
    	return 0; 
}
    
