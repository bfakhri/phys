#include <stdio.h> 
#include <stdlib.h>
#include <math.h> 
#include <mpi.h>


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
		}else if
	}

	unsigned int N, timeSteps;
    	N = atoi(argv[1]); 
	timeSteps = atoi(argv[2]);
	
	// Make more robust by checking oddity	
	unsigned int sectorSize = N/numNodes;	

	// Array to hold the values of cells for nodal subdivision
	double ** localMatrix = malloc(sizeof(double*)*sectorSize);
	for(unsigned int i=0; i<sectorSize; i++)
	{
		localMatrix[i] = malloc(sizeof(double)*N); 
		for(int j=0; j<N; j++)
			grid[i][j] = 0; 
	}
	

	

	// Reduce to get global max value in globMax 
	MPI_Allreduce((void*)&nodeMax, (void*)&globMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
	
	for(int t=0; t<timeSteps; t++)
	{
	}

	MPI_Finalize();    
    	return 0; 
}
    
