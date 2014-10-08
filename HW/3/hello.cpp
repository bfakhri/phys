#include <stdio.h> 
#include <stdlib.h>
#include <math.h> 
#include <mpi.h>
#include <string.h>

# define PI_CNST 3.14159265358979323846

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
			printf("Not enough or too many arguments\nExpecting 2");
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
	double ** localArray = malloc(sizeof(double*)*rowsPerSection);
	for(unsigned int i=0; i<rowsPerSection; i++)
	{
		localArray[i] = malloc(sizeof(double)*N); 
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
			localArray[r][c] = 0.5f;
		}
	}
	if(nodeRank == 0)
	{
		for(unsigned int c=0; c<N; c++)
		{
			localArray[0][c] = 0;
		}
	}
	if(nodeRank == (numNodes-1))
	{
		for(unsigned int c=0; c<N; c++)
		{
			localArray[rowsPerSection-1][c] = 5*sin(PI_CNST*pow((c/N),2)); 
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Take out unnecesary ones for top and bottom sections	
	double * lowerGhostRow = (double*)malloc(sizeof(double)*N);
	double * upperGhostRow = (double*)malloc(sizeof(double)*N);

	MPI_Request * sendDownReq = (MPI_Request*)malloc(sizeof(MPI_Request));
	MPI_Request * sendUpReq = (MPI_Request*)malloc(sizeof(MPI_Request));

	for(unsigned int t=0; t<timeSteps; t++)
	{
	
		// Send
		if(numNodes > 1)
		{
			// First Node
			if(nodeRank == 0)
			{
				// Receive from lower - sets up buffer
				MPI_Irecv((void*)lowerGhostRow, N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
				// Send down
				MPI_Isend((void*)localArray[rowsPerSection-1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
			
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
				// Lower
				for(unsigned int row = 0; row < rowsPerSection-2; row++)
				{
					memcpy((void*)localArray[row], (void*)tempArray[row], sizeof(double)*N);	
				}

				// Receive
				MPI_Wait(sendDownReq, MPI_STATUS_IGNORE);
				
				// Compute cells that depend on ghost rows
				tempArray[rowsPerSection-1][0] = (localArray[rowsPerSection-2][N-1]+localArray[rowsPerSection-2][0]+localArray[rowsPerSection-2][1]+
								localArray[rowsPerSection-1][N-1]+localArray[rowsPerSection-1][0]+localArray[rowsPerSection-1][1]+
								lowerGhostRow[N-1]+lowerGhostRow[0]+lowerGhostRow[1])/9; 
				for(unsigned int column =1; column < N-1; column++)
				{
					tempArray[rowsPerSection-1][column] = (localArray[rowsPerSection-2][column-1]+localArray[rowsPerSection-2][column]+localArray[rowsPerSection-2][column+1]+
								localArray[rowsPerSection-1][column-1]+localArray[rowsPerSection-1][column]+localArray[rowsPerSection-1][column+1]+
								lowerGhostRow[column-1]+lowerGhostRow[column]+lowerGhostRow[column+1])/9; 
				}
				tempArray[rowsPerSection-1][N-1] = (localArray[rowsPerSection-2][N-2]+localArray[rowsPerSection-2][N-1]+localArray[rowsPerSection-2][0]+
								localArray[rowsPerSection-1][N-2]+localArray[rowsPerSection-1][N-1]+localArray[rowsPerSection-1][0]+
								lowerGhostRow[N-2]+lowerGhostRow[N-1]+lowerGhostRow[0])/9; 
				
				// Copy last rows into local matrix
				memcpy((void*)localArray[rowsPerSection-2], tempArray[rowsPerSection-2], sizeof(double)*N);
				memcpy((void*)localArray[rowsPerSection-1], tempArray[rowsPerSection-1], sizeof(double)*N);
			}
			// Last Node
			else if(nodeRank == (numNodes-1))
			{
				// Receive from upper - sets up buffer
				MPI_Irecv((void*)upperGhostRow, N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendUpReq);
				// Send up
				MPI_Isend((void*)localArray[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendUpReq);
				
				// Compute nodes independent of ghost rows
				for(unsigned int row = 1; row < rowsPerSection-1; row++)
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
				for(unsigned int row = 3; row < rowsPerSection-2; row++)
				{
					memcpy((void*)localArray[row], (void*)tempArray[row], sizeof(double)*N);	
				}

				// Receive
				MPI_Wait(sendDownReq, MPI_STATUS_IGNORE);
				
				// Compute cells that depend on ghost rows
				// Upper 
				tempArray[0][0] = 		(upperGhostRow[N-1]+upperGhostRow[0]+upperGhostRow[1]+
								localArray[0][N-1]+localArray[0][0]+localArray[0][1]+
								localArray[1][N-1]+localArray[1][0]+localArray[1][1])/9; 
				for(unsigned int column =1; column < N-1; column++)
				{
					tempArray[rowsPerSection-1][column] = (upperGhostRow[column-1]+upperGhostRow[column]+upperGhostRow[column]+
							                     localArray[0][column-1]+localArray[0][column]+localArray[0][column]+
							                     localArray[1][column]+localArray[1][column]+localArray[1][column])/9; 
				}
				tempArray[0][N-1] =		(upperGhostRow[N-2]+upperGhostRow[N-1]+upperGhostRow[0]+     
						                localArray[0][N-2]+localArray[0][N-1]+localArray[0][0]+
						                localArray[1][N-2]+localArray[1][N-1]+localArray[1][0])/9; 


				// Copy last rows into local matrix
				memcpy((void*)localArray[0], tempArray[0], sizeof(double)*N);
				memcpy((void*)localArray[1], tempArray[1], sizeof(double)*N);
				memcpy((void*)localArray[2], tempArray[2], sizeof(double)*N);
			}
			// Middle Nodes
			else
			{
				// Receive from lower - sets up buffer
				MPI_Irecv((void*)lowerGhostRow, N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
				// Receive from upper - sets up buffer
				MPI_Irecv((void*)upperGhostRow, N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendUpReq);
				// Send down
				MPI_Isend((void*)localArray[rowsPerSection-1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
				// Send up
				MPI_Isend((void*)localArray[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendUpReq);
			
				// Compute nodes independent of ghost rows
				for(unsigned int row = 2; row < rowsPerSection-2; row++)
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
				for(unsigned int row = 3; row < rowsPerSection-3; row++)
				{
					memcpy((void*)localArray[row], (void*)tempArray[row], sizeof(double)*N);	
				}

				// Receive
				MPI_Wait(sendDownReq, MPI_STATUS_IGNORE);
				
				// Compute cells that depend on ghost rows
				// Upper
				tempArray[0][0] = 		(upperGhostRow[N-1]+upperGhostRow[0]+upperGhostRow[1]+
								localArray[0][N-1]+localArray[0][0]+localArray[0][1]+
								localArray[1][N-1]+localArray[1][0]+localArray[1][1])/9; 
				for(unsigned int column =1; column < N-1; column++)
				{
					tempArray[0][column] = (upperGhostRow[column-1]+upperGhostRow[column]+upperGhostRow[column]+
							                     localArray[0][column-1]+localArray[0][column]+localArray[0][column]+
							                     localArray[1][column]+localArray[1][column]+localArray[1][column])/9; 
				}
				tempArray[0][N-1] =		(upperGhostRow[N-2]+upperGhostRow[N-1]+upperGhostRow[0]+     
						                localArray[0][N-2]+localArray[0][N-1]+localArray[0][0]+
						                localArray[1][N-2]+localArray[1][N-1]+localArray[1][0])/9; 


				// Compute cells that depend on ghost rows
				// Lower
				tempArray[rowsPerSection-1][0] = (localArray[rowsPerSection-2][N-1]+localArray[rowsPerSection-2][0]+localArray[rowsPerSection-2][1]+
								localArray[rowsPerSection-1][N-1]+localArray[rowsPerSection-1][0]+localArray[rowsPerSection-1][1]+
								lowerGhostRow[N-1]+lowerGhostRow[0]+lowerGhostRow[1])/9; 
				for(unsigned int column =1; column < N-1; column++)
				{
					tempArray[rowsPerSection-1][column] = (localArray[rowsPerSection-2][column-1]+localArray[rowsPerSection-2][column]+localArray[rowsPerSection-2][column+1]+
								localArray[rowsPerSection-1][column-1]+localArray[rowsPerSection-1][column]+localArray[rowsPerSection-1][column+1]+
								lowerGhostRow[column-1]+lowerGhostRow[column]+lowerGhostRow[column+1])/9; 
				}
				tempArray[rowsPerSection-1][N-1] = (localArray[rowsPerSection-2][N-2]+localArray[rowsPerSection-2][N-1]+localArray[rowsPerSection-2][0]+
								localArray[rowsPerSection-1][N-2]+localArray[rowsPerSection-1][N-1]+localArray[rowsPerSection-1][0]+
								lowerGhostRow[N-2]+lowerGhostRow[N-1]+lowerGhostRow[0])/9; 
				// Copy last rows into local matrix
				memcpy((void*)localArray[0], tempArray[0], sizeof(double)*N);
				memcpy((void*)localArray[1], tempArray[1], sizeof(double)*N);
				memcpy((void*)localArray[2], tempArray[2], sizeof(double)*N);
				
				memcpy((void*)localArray[rowsPerSection-3], tempArray[rowsPerSection-3], sizeof(double)*N);
				memcpy((void*)localArray[rowsPerSection-2], tempArray[rowsPerSection-2], sizeof(double)*N);
				memcpy((void*)localArray[rowsPerSection-1], tempArray[rowsPerSection-1], sizeof(double)*N);
			}
		}

	}


	if(numNodes > 1)
	{
		double localSum = 0; 
		double globalSum = 0; 
		for(unsigned int row=0; row<rowsPerSection; row++)
			for(unsigned int column=0; column<N; column++)
			{
				localSum += localArray[row][column]; 
			}
		MPI_Reduce((void*)&localSum, (void*)&globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
		if(nodeRank == 0)
		{
			printf("Global Sum: %f", globalSum); 
		}
	}
	else
	{
		// Fill in lonely case
	}
	

	
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();    
    	return 0; 
}
    
