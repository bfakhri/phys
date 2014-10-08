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
	
	int threadRank, numThreads, numProcs;


	if(nodeRank == 0)
	{
		if(argc != 4)
		{
			 printf("Not enough or too many arguments\nExpecting 3");  
			return 1; 
		}
	}

	unsigned int N, timeSteps;
    int A; 
    N = atoi(argv[1]); 
    A = atoi(argv[2]);
	timeSteps = atoi(argv[3]);
/*
	if(nodeRank == 0){
		printf("Rough Diffuse using %d nodes and %d threads per node\nN = %d A = %d and timeSteps = %d\n\n", numNodes, omp_get_max_threads(), N, A, timeSteps);
	}
  */  										// A sector is a vertical subdivision of 
											// the diffusion grid. Each subdivision belongs 
	// Make more robust by checking oddity	// to a processing node (node, not core)
	unsigned int sectorSize = N/numNodes;	// sectorSize is the height of a sector 

	// Array to hold the values of cells for nodal subdivision
	double ** grid = malloc(sizeof(double*)*sectorSize);
	for(unsigned int i=0; i<sectorSize; i++){
		grid[i] = malloc(sizeof(double)*N); 
		for(int j=0; j<N; j++)
			grid[i][j] = 0; 
	}
	

	
	double samples = N*N/3; 
    unsigned int normalLimit = N/10; 

    double thetaStep = ((double)normalLimit)/samples;


    double xCord = 0; 
    double yCord = 0; 
    double magnitude = 0; 
    double theta = 0; 


	// HERE WE SET UP THE INITIAL CONDITION FOR EACH NODE //	

	int t;
    // Gives each core a different section of theta
	for(t=0; t<(unsigned int)samples; t++)
    {
        // ****** SPEEDUP ****** Simplify this math? 
        theta = t*thetaStep;  //*M_PI/180; 
        xCord = A*(cos(theta)+theta*sin(theta));        
        yCord = A*(sin(theta)-theta*cos(theta)); 
        magnitude = sin(theta)*sin(theta);
		// If the coordinates are in this sector we put the Z value in its place
    	if(((yCord+N/2) > nodeRank*sectorSize) && ((yCord+N/2) < (nodeRank+1)*sectorSize)) 
		{
			// If the coordinates are in the range of X (0-N/10) we put it in its place
			if((xCord+N/2 > 0) && (xCord+N/2 < N))
			{
				// Ensures we are only putting greatest calculated Z value for cell in cell
				if(magnitude > grid[(unsigned int)(yCord+N/2-nodeRank*sectorSize)][(unsigned int)(xCord+N/2)])
				{
					grid[(unsigned int)(yCord+N/2-nodeRank*sectorSize)][(unsigned int)(xCord+N/2)] = magnitude;    
				}
			}
		}
    }

	double globMax = 0; 		// Holds global maximum of Z
	double nodeMax = 0;			// Holds maximum Z for this node
	// Index 0 : x, 1 : y
	int * nodalCoordMax = malloc(sizeof(unsigned int)*2);	// Holds nodal x and y max cords
	int * globCoordMax = malloc(sizeof(unsigned int)*2);	// Holds global x and y max cords
	for(int a=0; a<2; a++){
		nodalCoordMax[a] = -1;
		globCoordMax[a] = -1;
	}

	// Speed up using openMP
	for(int y=0; y<sectorSize; y++){
		for(int x=0; x<N; x++){
			if(grid[y][x] > nodeMax){
				nodeMax = grid[y][x];
				nodalCoordMax[0] = x;
				nodalCoordMax[1] = y;
			}
		}
	}

	// Reduce to get global max value in globMax 
	MPI_Allreduce((void*)&nodeMax, (void*)&globMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
	
	if(!(globMax > nodeMax))
	{
		// Set maxCords
		globCoordMax[1] = nodalCoordMax[1]; 
		globCoordMax[0] = nodalCoordMax[0];
	}else{
		// Nulify maxCords
		globCoordMax[0] = -1; 	// Negative maxs means they aren't maxs
		globCoordMax[1] = -1; 	// Another node has max if it is negative
	}
	

	// HERE WE START TO DIFFUSE

	// Outer loop is the number of steps we take

	// Index 0 is down, Index 1 is UP
	MPI_Request * reqs = malloc(sizeof(MPI_Request)*2);		// 2 times for up and down
	// Make faster using openMP?
	// Index 0 is down, Index 1 is UP
	double ** rowCatcher = malloc(sizeof(double*)*2);		// Catches edge rows from other sectors
	for(int i=0; i<2; i++){
		rowCatcher[i] = malloc(sizeof(double)*N);	
		for(int j=0; j<N; j++){
			rowCatcher[i][j] = 0; 
		}
	}

	for(int t=0; t<timeSteps; t++)
	{
		// Send edge values so they will arrive before we need them
		// Non-blocking
		if((nodeRank > 0) && (nodeRank < numNodes-1)){
			// All nodes excluding first and last
			// Receive up
			MPI_Irecv((void*)rowCatcher[1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, &reqs[1]);
			// Receive down
			MPI_Irecv((void*)rowCatcher[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, &reqs[0]);
			// Send up
			MPI_Isend((void*)grid[sectorSize-1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, &reqs[1]);
			// Send down
			MPI_Isend((void*)grid[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, &reqs[0]);
		}else if(nodeRank > 0){
			// Last Node (rank == numNodes-1)
			// Receive down
			MPI_Irecv((void*)rowCatcher[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, &reqs[0]);
			// Send down
			MPI_Isend((void*)grid[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, &reqs[0]);
		}else if(numNodes > 1){
			// First node (rank == 0)
			// Receive up
			MPI_Irecv((void*)rowCatcher[1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, &reqs[1]);
			// Send up
			MPI_Isend((void*)grid[sectorSize-1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, &reqs[1]);
		}

		// TUNE THIS!!!!
		int y, x;
		for(y=1; y<sectorSize-1; y++)
		{
			for(x=1; x<N-1; x++)
			{
				grid[y][x] = 0.25*(grid[y][x+1]+grid[y][x-1]+grid[y+1][x]+grid[y-1][x]) - grid[y][x]; 
			}
			grid[y][0] = 0.25*(grid[y][x+1]+grid[y+1][x]+grid[y-1][x]) - grid[y][0]; 
			grid[y][N-1] = 0.25*(grid[y][N-2]+grid[y+1][N-1]+grid[y-1][N-1]) - grid[y][N-1]; 
			
		}  		                         

		// Wait for edge values to come in	
		if((nodeRank > 0) && (nodeRank < numNodes-1)){
			// All nodes excluding first and last
			MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
			MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
		}else if(nodeRank > 0){
			// Last Node (rank == numNodes-1)
			MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
		}else if(numNodes > 1){
			// First node (rank == 0)
			MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
		}

		for(x=1; x<N-1; x++)
		{
			grid[0][x] = 0.25*(grid[0][x+1]+grid[1][x]+grid[0][x-1]+rowCatcher[0][x]) - grid[0][x]; 
			grid[sectorSize-1][x] = 0.25*(grid[sectorSize-1][x+1]+grid[sectorSize-1][x-1]+grid[sectorSize-2][x]+rowCatcher[1][x]) - grid[sectorSize-1][x]; 

		}
		grid[0][0] = 0.25*(grid[1][0]+grid[0][1]+rowCatcher[0][0]) - grid[0][0]; 
		grid[0][N-1] = 0.25*(grid[1][N-1]+grid[0][N-2]+rowCatcher[0][N-1]) - grid[0][N-1]; 
		grid[sectorSize-1][0] = 0.25*(grid[sectorSize-1][1]+grid[sectorSize-2][0]+rowCatcher[1][0]) - grid[sectorSize-1][0]; 
		grid[sectorSize-1][N-1] = 0.25*(grid[sectorSize-1][N-2]+grid[sectorSize-2][N-1]+rowCatcher[1][N-1]) - grid[sectorSize-1][N-1];
 
		// Speed up using openMP
		for(int y=0; y<sectorSize; y++){
			for(int x=0; x<N; x++){
				if(grid[y][x] > nodeMax){
					nodeMax = grid[y][x];
					nodalCoordMax[0] = x;
					nodalCoordMax[1] = y;
				}
			}
		}
		// Reduce to get global max value in globMax 
		MPI_Allreduce((void*)&nodeMax, (void*)&globMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); 
		if(!(globMax > nodeMax))
		{
			// Set maxCords
			globCoordMax[1] = nodalCoordMax[1]; 
			globCoordMax[0] = nodalCoordMax[0]-N/2;
			// Print Max Values
			//printf("Time: %d\tGlobal Max: %f\t At location X=%d, Y=%d\n", t, globMax, globCoordMax[0]-N/2, globCoordMax[1]+nodeRank*sectorSize-N/2); 
		}else{
			// Nulify maxCords
			globCoordMax[0] = -1; 	// Negative maxs means they aren't maxs
			globCoordMax[1] = -1; 	// Another node has max if it is negative
		}

		


	}

	if(globCoordMax[0] != -1){
		printf("Global Max: %f\t At location X=%d, Y=%d\n", globMax, globCoordMax[0]-N/2, globCoordMax[1]+nodeRank*sectorSize-N/2); 
	}
		
	MPI_Finalize();    
    return 0; 
}
    
