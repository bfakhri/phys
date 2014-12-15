// Main

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

#include "mass.h"
#include "mather.h"
#include "debug.h"

#define DEF_TIME_STEP 1
#define DEF_SIM_STEPS 100
#define DEF_STEP_PER_OUTPUT 1
#define DEF_TOTAL_OBJECTS 2

using namespace std; 


int main(int argc, char ** argv)
{
	// Simulation parameter variables
	double TIME_STEP = DEF_TIME_STEP;
	uint32_t SIM_STEPS = DEF_SIM_STEPS;
	uint32_t STEP_PER_OUTPUT = DEF_STEP_PER_OUTPUT;  
	uint32_t TOTAL_OBJECTS = DEF_TOTAL_OBJECTS; 

	// Custom simulation parameters 
	if(argc > 1)
	{
		if(argc != 4){
			cout << endl << "ERROR, incorrect number of arguments" << endl; 
			return -1; 
		}else{
			SIM_STEPS = atoi(argv[1]); 
			TIME_STEP = atoi(argv[2]); 
			STEP_PER_OUTPUT = atoi(argv[3]); 	
		}
	}
	cout << "Simulation: " << endl << "Number of steps = " << SIM_STEPS 
		<< endl << "Size of time step (seconds) = " << TIME_STEP << endl 
		<< "Steps per output = " << STEP_PER_OUTPUT << endl; 


	// Initialize constants like G
	initConsts(); 

	// Initialize output 
	initOutput(); 

	// Holds all of the masses 
	vector<Mass> massVector; 

	// Earth/Moon sim
	Mass earth, moon; 
	earth.setMass(scientificNotation(5.9736, 24)); 
	earth.setName("Earth"); 
	moon.setMass(scientificNotation(7.349, 22)); 
	moon.setName("Moon"); 
	moon.setPos(scientificNotation(3.626, 8), 0, 0); 
	moon.setVelocity(0, 1023, 0); 

	massVector.push_back(earth);
	massVector.push_back(moon);

	// Export object count to output	
	outputObjectCount((uint32_t)massVector.size()); 
	// Export masses to output
	for(int i=0; i<massVector.size(); i++){
		outputObjectMass(massVector[i].getMass()); 
	}
	// Export number of simulation steps
	outputSimSteps(SIM_STEPS); 	


	// MAIN SIMULATION LOOP
	for(int t=0; t<SIM_STEPS; t++)
	{	
		// Influences
		for(int i=0; i<massVector.size(); i++)
		{
			massVector[i].resetForces(); 
			for(int j=0; j<massVector.size(); j++)
			{
				if(i != j)
				{
					massVector[i].influence(massVector[j]); 
				}
			}

		}
		
		// Update position
		for(int i=0; i<massVector.size(); i++)
		{
			// General case output
			if(c%STEP_PER_OUTPUT == 0){
				//printF(massVector[1]);
				//printVel(massVector[1]);  
				printTime(t*TIME_STEP);
				//printVel(massVector[1]);  
				printPos(massVector[1]); 
				printDist(massVector[0], massVector[1]); 
				
			}
			// Export coordinates of all objects to output
			outputFrames(massVector[i].getPos().x, massVector[i].getPos().y);

			massVector[i].updateVelAndPos(TIME_STEP); 
			massVector[i].resetForces();  
		}	
	}

	// Close output file
	outputClose(); 

	return 0;
}
