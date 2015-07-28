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
#include "output.h"

#define DEF_TIME_STEP 1
#define DEF_SIM_STEPS 100
#define DEF_STEP_PER_OUTPUT 1

using namespace std; 


int main(int argc, char ** argv)
{
	// Simulation parameter variables
	double TIME_STEP = DEF_TIME_STEP;
	uint32_t SIM_STEPS = DEF_SIM_STEPS;
	uint32_t STEP_PER_OUTPUT = DEF_STEP_PER_OUTPUT;  

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
	Mass solarSystem[9]; 
	solarSystem[0].setMass(scientificNotation(1.99, 30)); 
	solarSystem[0].setPos(scientificNotation(0, 1), 0, 0); 
	solarSystem[0].setVelocity(0, scientificNotation(0, 1), 0);
	solarSystem[1].setMass(scientificNotation(3.301, 23)); 
	solarSystem[1].setPos(scientificNotation(6.9, 10), 0, 0); 
	solarSystem[1].setVelocity(0, scientificNotation(3.94, 4), 0);
	solarSystem[2].setMass(scientificNotation(4.867, 24)); 
	solarSystem[2].setPos(scientificNotation(1.09, 11), 0, 0); 
	solarSystem[2].setVelocity(0, scientificNotation(3.48, 4), 0);
	solarSystem[3].setMass(scientificNotation(5.97, 24)); 
	solarSystem[3].setPos(scientificNotation(1.47, 11), 0, 0); 
	solarSystem[3].setVelocity(0, scientificNotation(3.026, 4), 0);
	solarSystem[4].setMass(scientificNotation(6.42, 23)); 
	solarSystem[4].setPos(scientificNotation(2.83, 11), 0, 0); 
	solarSystem[4].setVelocity(0, scientificNotation(2.65, 4), 0);
	solarSystem[5].setMass(scientificNotation(1.898, 27)); 
	solarSystem[5].setPos(scientificNotation(7.95, 11), 0, 0); 
	solarSystem[5].setVelocity(0, scientificNotation(1.28, 4), 0);
	solarSystem[6].setMass(scientificNotation(5.68, 26)); 
	solarSystem[6].setPos(scientificNotation(1.49, 12), 0, 0); 
	solarSystem[6].setVelocity(0, scientificNotation(9.216, 3), 0);
	solarSystem[7].setMass(scientificNotation(8.68, 25)); 
	solarSystem[7].setPos(scientificNotation(2.99, 12), 0, 0); 
	solarSystem[7].setVelocity(0, scientificNotation(6.51, 3), 0);
	solarSystem[8].setMass(scientificNotation(1.02, 26)); 
	solarSystem[8].setPos(scientificNotation(4.49, 12), 0, 0); 
	solarSystem[8].setVelocity(0, scientificNotation(5.44, 3), 0);

 	for(int i=0; i<8; i++){
		massVector.push_back(solarSystem[i]); 
	}

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
		
		// Moon output
		if(t%STEP_PER_OUTPUT == 0){
			//printF(massVector[1]);
			//printVel(massVector[1]);  
			//printTime(t*TIME_STEP);
			//printVel(massVector[1]);  
			printPos(massVector[1]);
			cout << endl; 
			printPos(massVector[2]); 
			printDist(massVector[0], massVector[1]); 
			
		}

		// Update position
		for(int i=0; i<massVector.size(); i++)
		{
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
