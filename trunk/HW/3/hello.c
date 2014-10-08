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
	
	double ** tempArray = (double**)malloc(sizeof(double*)*rowsPerSection);
	for(unsigned int i=0; i<N; i++)
	{
		tempArray[i] = (double*)malloc(sizeof(double)*N);
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
			
				// COULD MAKE IT FASTER BY TAKING ITERATIONS THAT INCLUDE THE TOP ROW OUT OF THE LOOP AND 
				// TAKE INTO ACCOUNT THAT THE CELLS AT THE TOP ARE 0 and thus do not matter to the average
					
				// Compute nodes independent of ghost rows
				for(unsigned int row = 1; row < rowsPerSection-2; row++)
				{
					// Left Column 
					tempArray[row][0] = (localArray[row-1][N-1]+localArray[row-1][0]+localArray[row-1][1]+
								localArray[row][N-1]+localArray[row][0]+localArray[row][1]+
								localArray[row+1][N-1]+localArray[row+1][0]+localArray[row+1][1])/9; 

					// Middle rows
					for(unsigned int column = 1; column < N-1; column++)
					{	
						tempArray[row][column] = (localArray[row-1][column-1]+localArray[row-1][column]+localArray[row-1][column+1]+
									localArray[row][column-1]+localArray[row][column]+localArray[row][column+1]+
									localArray[row+1][column-1]+localArray[row+1][column]+localArray[row+1][column+1])/9; 
					}	
					
					// Right Column
					tempArray[row][N-1] = (localArray[row-1][N-2]+localArray[row-1][N-1]+localArray[row-1][0]+
								localArray[row][N-2]+localArray[row][N-1]+localArray[row][0]+
								localArray[row+1][N-2]+localArray[row+1][N-1]+localArray[row+1][0])/9; 

				}	

				// Copy rows over
				for(unsigned int row = 0; row < rowsPerSection-1; row++)
				{
					
				}
				memcpy(tempArray[row], localArray[row], sizeof(double)*N); 	
				// ////////

				// Receive
				MPI_Wait(sendDownReq, MPI_STATUS_IGNORE);
				
				// Compute cells that depend on ghost rows
				tempArray[rowsPerSection-1][0] = (localArray[rowsPerSection-2][N-1]+localArray[rowsPerSection-2][0]+localArray[rowsPerSection-2][1]+
								localArray[rowsPerSection-1][N-1]+localArray[rowsPerSection-1][0]+localArray[rowsPerSection-1][1]+
								lowerGhostRow[N-1]+lowerGhostRow[0]+lowerGhostRow[1])/9; 
				for(unsigned int column =1; column < N-1; column++)
				{
					tempArray[rowPerSection-1][column] = (localArray[row-1][column-1]+localArray[row-1][column]+localArray[row-1][column+1]+
								localArray[row][column-1]+localArray[row][column]+localArray[row][column+1]+
								lowerGhostRow[column-1]+lowerGhostRow[column]+lowerGhostRow[column+1])/9; 
				}
				tempArray[rowsPerSection-1][N-1] = (localArray[rowsPerSection-2][N-2]+localArray[rowsPerSection-2][N-1]+localArray[rowsPerSection-2][0]+
								localArray[rowsPerSection-1][N-2]+localArray[rowsPerSection-1][N-1]+localArray[rowsPerSection-1][0]+
								lowerGhostRow[N-2]+lowerGhostRow[N-1]+lowerGhostRow[0])/9; 

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
		for(unsigned int row = 1; row < rowsPerSection-1; row++)
		{
			for(unsigned int column = 1; column < N-1; column++)
			{
				tempArray[row][column] = (localArray[row-1][column-1]+localArray[row-1][column]+localArray[row-1][column+1]+
							localArray[row][column-1]+localArray[row][column]+localArray[row][column+1]+
							localArray[row+1][column-1]+localArray[row+1][column]+localArray[row+1][column+1])/9; 
			}	
			
			// Wrap Left	- CORRECT THIS TO WRAP AROUND
			tempArray[row][column] = (localArray[row-1][column-1]+localArray[row-1][column]+localArray[row-1][column+1]+
						localArray[row][column-1]+localArray[row][column]+localArray[row][column+1]+
						localArray[row+1][column-1]+localArray[row+1][column]+localArray[row+1][column+1])/9; 
			// Wrap Right    - CORRECT THIS TO WRAP AROUND
			tempArray[row][column] = (localArray[row-1][column-1]+localArray[row-1][column]+localArray[row-1][column+1]+
						localArray[row][column-1]+localArray[row][column]+localArray[row][column+1]+
						localArray[row+1][column-1]+localArray[row+1][column]+localArray[row+1][column+1])/9; 

		}	
		

		// COPY TEMP ARRAY TO ORIGINAL ARRAY
		// HERE

		// Receive	
		// Wait for edge values to come in	
		if((nodeRank > 0) && (nodeRank < numNodes-1)){
			// All nodes excluding first and last
			MPI_Wait(sendUpReq, MPI_STATUS_IGNORE);
			MPI_Wait(sendDownReq, MPI_STATUS_IGNORE);
		}else if(nodeRank > 0){
			// Last Node (rank == numNodes-1)
			MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
		}else if(numNodes > 1){
		}else
		{
			// CASE FOR ONLY 1 NODE
			// 
		}
	
		// CALC FOR MISSING ROWS 
		if(nodeRank
		for(unsigned int column = 0; column < N; column++)
		{
			localArray[1][column] = 
	
	}


	// Reduce to get global max value in globMax 
	MPI_Allreduce((void*)&nodeMax, (void*)&globMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
	
	for(int t=0; t<timeSteps; t++)
	{
	}

	MPI_Finalize();    
    	return 0; 
}
    
