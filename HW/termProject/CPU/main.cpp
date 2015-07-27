// Main

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <cstdlib> 

#include "mass.h"
#include "mather.h"
#include "debug.h"
#include "output.h"

#define DEF_TIME_STEP 1
#define DEF_SIM_STEPS 100
#define DEF_STEP_PER_OUTPUT 1
#define DEF_PROGRESS_OUT 1
#define DEF_RESULTS_OUT 1
#define DEF_CPU_PARALLEL 1
#define DEF_N 120

using namespace std; 


int main(int argc, char ** argv)
{
	cout << endl << endl; 
	// Simulation parameter variables
	double TIME_STEP = DEF_TIME_STEP;
	uint64_t SIM_STEPS = DEF_SIM_STEPS;
	uint64_t STEP_PER_OUTPUT = DEF_STEP_PER_OUTPUT;  
	int PROGRESS_OUT = DEF_PROGRESS_OUT; 
	int RESULTS_OUT = DEF_RESULTS_OUT; 
	int CPU_PARALLEL = DEF_CPU_PARALLEL; 
	uint64_t N = DEF_N; 

	// Custom simulation parameters 
	if(argc > 1)
	{
		if(argc != 8){
			cout << endl << "ERROR, incorrect number of arguments" << endl; 
			return -1; 
		}else{
			SIM_STEPS = atoi(argv[1]); 
			TIME_STEP = atoi(argv[2]); 
			STEP_PER_OUTPUT = atoi(argv[3]); 	
			PROGRESS_OUT = atoi(argv[4]); 
			RESULTS_OUT = atoi(argv[5]); 
			CPU_PARALLEL = atoi(argv[6]); 
			N = atoi(argv[7]); 
		}
	}
	cout << "Simulation: " << endl << "Number of steps = " << SIM_STEPS 
		<< endl << "Size of time step (seconds) = " << TIME_STEP << endl 
		<< "Steps per output = " << STEP_PER_OUTPUT << endl
		<< "Progress out?  = " << PROGRESS_OUT << endl 
		<< "Results out? = " << RESULTS_OUT << endl
		<< "CPU Parallel? = " << CPU_PARALLEL << endl
		<< "N (masses)? = " << N << endl; 


	// Initialize constants like G
	initConsts(); 

	if(PROGRESS_OUT){
		// Initialize output 
		initOutput(); 
	}

	// Holds all of the masses 
	vector<Mass> massVector; 

	for(uint64_t i=0; i<N; i++){
		double rMass;
		cartesian rPos; 
		cartesian rVel; 
		rMass = scientificNotation(rand()%10+1, rand()%30+1);
		uint32_t dirX = rand()%2; 
		uint32_t dirY = rand()%2; 
		if(dirX == 0)
			rPos.x = scientificNotation(rand()%9+1+i, rand()%3+10);
		else
			rPos.x = -1*scientificNotation(rand()%9+1+i, rand()%3+10);
		if(dirY == 0)
			rPos.y = scientificNotation(rand()%9+1+i, rand()%3+10);
		else
			rPos.y = -1*scientificNotation(rand()%9+1+i, rand()%3+10);

		rPos.z = scientificNotation(rand()%9+1+i, rand()%3+10);
		rVel.x = scientificNotation(rand()%9+1, rand()%3+2);
		rVel.y = scientificNotation(rand()%9+1, rand()%3+2);
		rVel.z = scientificNotation(rand()%9+1, rand()%3+2);
		massVector.push_back(*(new Mass(rMass, rPos, rVel))); 
	}

	if(PROGRESS_OUT){
		// Export object count to output	
		outputObjectCount((uint64_t)massVector.size()); 
		// Export masses to output
		for(uint64_t i=0; i<massVector.size(); i++){
			outputObjectMass(massVector[i].getMass()); 
		}
		// Export number of simulation steps
		outputSimSteps(SIM_STEPS); 	
	}

	// MAIN SIMULATION LOOP
	for(uint64_t t=0; t<SIM_STEPS; t++)
	{	
		// Influences
		#pragma omp parallel for if(CPU_PARALLEL) schedule(guided)
		for(uint64_t i=0; i<massVector.size(); i++)
		{
			massVector[i].resetForces(); 
			for(uint64_t j=0; j<massVector.size(); j++)
			{
				if(i != j)
				{
					massVector[i].influence(massVector[j]); 
				}
			}

		}
		
		// Moon some 
		if(PROGRESS_OUT){
			if(t%STEP_PER_OUTPUT == 0){
				printTime(t*TIME_STEP);
				printPos(massVector[1]);
				printDist(massVector[0], massVector[1]); 
				
			}
		}

		// Update position
		#pragma omp parallel for if(CPU_PARALLEL) schedule(guided)
		for(uint64_t i=0; i<massVector.size(); i++)
		{
			if(PROGRESS_OUT){
				// Export coordinates of all objects to output
				outputFrames(massVector[i].getPos().x, massVector[i].getPos().y);
			}

			massVector[i].updateVelAndPos(TIME_STEP); 
			massVector[i].resetForces();  
		}	
	}

	if(PROGRESS_OUT){
		// Close output file
		outputClose(); 
	}

	if(RESULTS_OUT){
		// Prints last positions of masses to file
		outputResults(massVector); 
	}


	cout << endl << endl; 

	return 0;
}
