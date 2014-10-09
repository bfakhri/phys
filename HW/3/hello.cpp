#include <iostream> 
#include <stdlib.h>
#include <math.h> 
#include <mpi.h>
#include <string.h>

# define PI_CNST 3.14159265358979323846
using namespace std;


int main(int argc, char *argv[])
{
   	int nodeRank, numNodes, length; 
	char name[128];  
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&nodeRank);
	MPI_Comm_size(MPI_COMM_WORLD,&numNodes);
	MPI_Get_processor_name(name, &length); 


	if(nodeRank == 0)
	{
		if(argc != 3)
		{
			cout << "Not enough or too many arguments" << endl << "Expecting 2" << endl;
			return 1; 
		}
	}

	unsigned int N, timeSteps;
    	N = atoi(argv[1]); 
	timeSteps = atoi(argv[2]);

	//cout<<"Rank: "<< nodeRank << "\tName: " << name << endl;

	// For output purposes
	if(nodeRank == 0)
		cout<<"numNodes = "<<numNodes<<"\tN = "<<N<<"\ttimeSteps = "<<timeSteps<<endl;

	// Find section sizes	
	unsigned int rowsPerSection = (N)/numNodes;
	unsigned int remainder = (N)%numNodes;
	if((nodeRank+1) <= remainder)
		rowsPerSection++;
	


	// Array to hold the values of cells for nodal subdivision
	double ** localArray = new double*[rowsPerSection];
	for(unsigned int i=0; i<rowsPerSection; i++)
		localArray[i] = new double[N]; 

	// Array to hold values of next iteration	
	double ** tempArray = new double*[rowsPerSection];
	for(unsigned int i=0; i<rowsPerSection; i++)
		tempArray[i] = new double[N]; 
	

	// Fill in the matrix with initial values
	for(unsigned int r=0; r<rowsPerSection; r++)
	{
		for(unsigned int c=0; c<N; c++)
		{
			localArray[r][c] = 0.5f;
		}
	}

	// Fill in top row
	if(nodeRank == 0)
	{
		for(unsigned int c=0; c<N; c++)
		{
			localArray[0][c] = 0;
		}
	}
	
	// Fill in bottom row
	if(nodeRank == (numNodes-1))
	{
		for(unsigned int c=0; c<N; c++)
		{
			localArray[rowsPerSection-1][c] = 5*sin(PI_CNST*pow((((double)(c))/N),2)); 
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	double timeElapsed = 0;
	if(nodeRank == 0)
		timeElapsed = MPI_Wtime();

	// Take out unnecesary ones for top and bottom sections	
	double * lowerGhostRow = new double[N]; 
	double * upperGhostRow = new double[N];

	MPI_Request * sendDownReq = new MPI_Request;
	MPI_Request * sendUpReq = new MPI_Request;
	MPI_Request * bogusReq = new MPI_Request;

	for(unsigned int t=0; t<timeSteps; t++)
	{
		if(numNodes > 1)
		{
			// First Node
			if(nodeRank == 0)
			{
				// Receive from lower - sets up buffer
				MPI_Irecv((void*)lowerGhostRow, N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendDownReq);
				// Send down
				MPI_Isend((void*)localArray[rowsPerSection-1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, bogusReq);
				
				// Compute nodes independent of ghost rows
				for(unsigned int row = 1; row < rowsPerSection-1; row++)
				{

					// Middle rows
					for(unsigned int column = 0; column < N; column++)
					{	
						unsigned int lC;
						if(((int)column-1) < 0)
							lC = N-1; 
						else
							lC = column-1;
						unsigned int rC; 
						if(((int)column+1) >= N)
							rC = 0; 
						else
							rC = column+1;
						tempArray[row][column] = (localArray[row-1][lC]+localArray[row-1][column]+localArray[row-1][rC]+
									localArray[row][lC]+localArray[row][column]+localArray[row][rC]+
									localArray[row+1][lC]+localArray[row+1][column]+localArray[row+1][rC])/9; 
					}	
				}	

				// Copy rows over
				for(unsigned int row = 1; row < rowsPerSection-2; row++)
				{
					memcpy((void*)localArray[row], (void*)tempArray[row], sizeof(double)*N);	
				}

				// Receive
				MPI_Wait(sendDownReq, MPI_STATUS_IGNORE);
				
				// Compute cells that depend on ghost rows
				for(unsigned int column = 0; column < N; column++)
				{
					unsigned int lC;
					if(((int)column-1) < 0)
						lC = N-1; 
					else
						lC = column-1;
					unsigned int rC; 
					if(((int)column+1) >= N)
						rC = 0; 
					else
						rC = column+1;
					unsigned int row = rowsPerSection-1;
					tempArray[row][column] = (localArray[row-1][lC]+localArray[row-1][column]+localArray[row-1][rC]+
								localArray[row][lC]+localArray[row][column]+localArray[row][rC]+
								lowerGhostRow[lC]+lowerGhostRow[column]+lowerGhostRow[rC])/9; 
				}
				
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
				MPI_Isend((void*)localArray[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, bogusReq);
				
				// Compute nodes independent of ghost rows
				for(unsigned int row = 1; row < rowsPerSection-1; row++)
				{
					// Middle rows
					for(unsigned int column = 0; column < N; column++)
					{	
						unsigned int lC;
						if(((int)column-1) < 0)
							lC = N-1; 
						else
							lC = column-1;
						unsigned int rC; 
						if(((int)column+1) >= N)
							rC = 0; 
						else
							rC = column+1;
						tempArray[row][column] = (localArray[row-1][lC]+localArray[row-1][column]+localArray[row-1][rC]+
									localArray[row][lC]+localArray[row][column]+localArray[row][rC]+
									localArray[row+1][lC]+localArray[row+1][column]+localArray[row+1][rC])/9; 
					}	
				}	

				// Copy rows over
				for(unsigned int row = 2; row < rowsPerSection-1; row++)
				{
					memcpy((void*)localArray[row], (void*)tempArray[row], sizeof(double)*N);	
				}

				// Receive
				MPI_Wait(sendUpReq, MPI_STATUS_IGNORE);
				
				// Compute cells that depend on ghost rows
				// Upper 
				for(unsigned int column =1; column < N-1; column++)
				{
					unsigned int lC;
					if(((int)column-1) < 0)
						lC = N-1; 
					else
						lC = column-1;
					unsigned int rC; 
					if(((int)column+1) >= N)
						rC = 0; 
					else
						rC = column+1;
					
					unsigned int row = 0;
					tempArray[row][column] = (upperGhostRow[lC]+upperGhostRow[column]+upperGhostRow[rC]+
								localArray[row][lC]+localArray[row][column]+localArray[row][rC]+
								localArray[row+1][lC]+localArray[row+1][column]+localArray[row+1][rC])/9; 
				}
				
				// Copy last rows into local matrix
				memcpy((void*)localArray[0], tempArray[0], sizeof(double)*N);
				memcpy((void*)localArray[1], tempArray[1], sizeof(double)*N);
			}
			// Middle Nodes
			else
			{

				// Receive from lower - sets up buffer
				MPI_Irecv((void*)lowerGhostRow, N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, sendUpReq);
				// Receive from upper - sets up buffer
				MPI_Irecv((void*)upperGhostRow, N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, sendDownReq);
				// Send down
				MPI_Isend((void*)localArray[rowsPerSection-1], N, MPI_DOUBLE, nodeRank+1, t, MPI_COMM_WORLD, bogusReq);
				// Send up
				MPI_Isend((void*)localArray[0], N, MPI_DOUBLE, nodeRank-1, t, MPI_COMM_WORLD, bogusReq);
			
				// Compute nodes independent of ghost rows
				for(unsigned int row = 1; row < rowsPerSection-1; row++)
				{
					// Middle rows
					for(unsigned int column = 0; column < N; column++)
					{	
						unsigned int lC;
						if(((int)column-1) < 0)
							lC = N-1; 
						else
							lC = column-1;
						unsigned int rC; 
						if(((int)column+1) >= N)
							rC = 0; 
						else
							rC = column+1;
						tempArray[row][column] = (localArray[row-1][lC]+localArray[row-1][column]+localArray[row-1][rC]+
									localArray[row][lC]+localArray[row][column]+localArray[row][rC]+
									localArray[row+1][lC]+localArray[row+1][column]+localArray[row+1][rC])/9; 
					}	
				}	

				// Copy rows over
				for(unsigned int row = 2; row < rowsPerSection-2; row++)
				{
					memcpy((void*)localArray[row], (void*)tempArray[row], sizeof(double)*N);	
				}

				// Receive
				MPI_Wait(sendDownReq, MPI_STATUS_IGNORE);
				MPI_Wait(sendUpReq, MPI_STATUS_IGNORE); 
				
				// Compute cells that depend on ghost rows
				// Upper
				for(unsigned int column =1; column < N-1; column++)
				{
					unsigned int lC;
					if(((int)column-1) < 0)
						lC = N-1; 
					else
						lC = column-1;
					unsigned int rC; 
					if(((int)column+1) >= N)
						rC = 0; 
					else
						rC = column+1;
					unsigned int row = 0; 
					tempArray[row][column] = (upperGhostRow[lC]+upperGhostRow[column]+upperGhostRow[rC]+
								localArray[row][lC]+localArray[row][column]+localArray[row][rC]+
								localArray[row+1][lC]+localArray[row+1][column]+localArray[row+1][rC])/9; 
				}

				// Compute cells that depend on ghost rows
				// Lower

				for(unsigned int column =1; column < N-1; column++)
				{
					unsigned int lC;
					if(((int)column-1) < 0)
						lC = N-1; 
					else
						lC = column-1;
					unsigned int rC; 
					if(((int)column+1) >= N)
						rC = 0; 
					else
						rC = column+1;
					unsigned int row = 0; 
					tempArray[row][column] = (localArray[row-1][lC]+localArray[row-1][column]+localArray[row-1][rC]+
								localArray[row][lC]+localArray[row][column]+localArray[row][rC]+
								lowerGhostRow[lC]+lowerGhostRow[column]+lowerGhostRow[rC])/9; 
				}
				// Copy last rows into local matrix
				memcpy((void*)localArray[0], tempArray[0], sizeof(double)*N);
				memcpy((void*)localArray[1], tempArray[1], sizeof(double)*N);
				
				memcpy((void*)localArray[rowsPerSection-2], tempArray[rowsPerSection-2], sizeof(double)*N);
				memcpy((void*)localArray[rowsPerSection-1], tempArray[rowsPerSection-1], sizeof(double)*N);
			}
		}else
		// Single Node
		{
			// No Comm necessary for single node
			
			for(unsigned int row = 1; row < rowsPerSection-1; row++)
			{
				for(unsigned int column = 0; column < N; column++)
				{
					unsigned int lC;
					if(((int)column-1) < 0)
						lC = N-1; 
					else
						lC = column-1;
					unsigned int rC; 
					if(((int)column+1) >= N)
						rC = 0; 
					else
						rC = column+1;
					tempArray[row][column] = (localArray[row-1][lC]+localArray[row-1][column]+localArray[row-1][rC]+
								localArray[row][lC]+localArray[row][column]+localArray[row][rC]+
								localArray[row+1][lC]+localArray[row+1][column]+localArray[row+1][rC])/9; 
				}
			}
			
			// Copy rows back into original matrix
			for(unsigned int row = 1; row < rowsPerSection-1; row++)	
				memcpy((void*)localArray[row], (void*)tempArray[row], sizeof(double)*N);

			/*
			for(int i=0; i<N; i++)
			{
				for(int j=0; j<N; j++)
				{
					cout << localArray[i][j] << "  ";
				}
				cout << endl;
			}
			*/
		}
	}


	// Compute verification sum
	double localSum = 0; 
	double globalSum = 0;
	unsigned int rps = (N)/numNodes;
	remainder = (N)%numNodes;
	unsigned int offset = 0; 
	
	if(remainder != 0)
	{
		if(nodeRank > remainder)
		{
			offset += (remainder)*(rps+1)+(nodeRank-remainder)*rps; 
		}
		else
		{
			offset += nodeRank*rps; 
		}
	}
	else
	{
		offset += nodeRank*rps; 
	}

	if(numNodes > 1)
	{
		for(unsigned int i=0; i<rowsPerSection; i++)
		{
			localSum += localArray[i][i+offset]; 
		}
		
		MPI_Reduce((void*)&localSum, (void*)&globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
	}
	else
	{
		// 1 node case
		for(unsigned int rc = 0; rc < rowsPerSection; rc++)
			globalSum += localArray[rc][rc]; 
	}
	
	
	double timeElapsed2 = 0; 
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(nodeRank == 0)
		timeElapsed2 = MPI_Wtime();

	if(nodeRank == 0)
	{
		cout<<"Rank: " << nodeRank << "\tLocal Sum: " << localSum << "\tGlobal Sum: "<<globalSum<<"\tTime Elapsed: "<<timeElapsed2-timeElapsed<<endl; 
	}



	// Take out unnecesary ones for top and bottom sections	
	for(unsigned int r = 0; r<rowsPerSection; r++)
	{
		delete[] localArray[r];
		delete[] tempArray[r];
	}

	delete[] localArray;
	delete[] tempArray;

	delete[] lowerGhostRow; 
	delete[] upperGhostRow;

	delete sendDownReq;
	delete sendUpReq;
	delete bogusReq; 

	MPI_Finalize();    
    	return 0; 
}
    
