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

using namespace std; 


int main(int argc, char ** argv)
{
	double TIME_STEP = DEF_TIME_STEP;
	uint32_t SIM_STEPS = DEF_SIM_STEPS;
	uint32_t STEP_PER_OUTPUT = DEF_STEP_PER_OUTPUT;  

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

	// Ouput file stuff
	FILE * vidFile;
	vidFile = fopen("vidFile.bin", "wb"); 

	initConsts(); 

	vector<Mass> massVector; 

	Mass earth, moon; 
	earth.setMass(scientificNotation(5.9736, 24)); 
	earth.setName("Earth"); 
	moon.setMass(scientificNotation(7.349, 22)); 
	moon.setName("Moon"); 
	moon.setPos(scientificNotation(3.626, 8), 0, 0); 
	//moon.setVelocity(0, 0, 1076); 
	//moon.setVelocity(0, 1076000, 0); 
	moon.setVelocity(0, 1023, 0); 
	//moon.setVelocity(0, 0, 800); 

	massVector.push_back(earth);
	massVector.push_back(moon);
	

	cout << "G:\t" << G << endl; 
	cout << "Earth Mass:\t" << earth.getMass() << endl; 
	cout << "Moon Mass:\t" << moon.getMass() << endl; 
	
	
	// Output file stuff
	uint32_t numObjects = 2; 
	fwrite ( (const void*)(&numObjects), 4, 1, vidFile );
	fwrite ( (const void*)(&SIM_STEPS), 4, 1, vidFile );


	// Simulation loop
	for(int c=0; c<SIM_STEPS; c++)
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
				printTime(c*TIME_STEP);
				//printVel(massVector[1]);  
				printPos(massVector[1]); 
				printDist(massVector[0], massVector[1]); 
				
			}
			// For output
			double tX, tY; 
			tX = massVector[i].getPos().x;
			tY = massVector[i].getPos().y;
			fwrite((const void *)(&tX), 8, 1, vidFile); 
			fwrite((const void *)(&tY), 8, 1, vidFile); 
			massVector[i].updateVelAndPos(TIME_STEP); 
			massVector[i].resetForces();  
		}	
	}

	return 0;
}
