// Main

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "mass.h"
#include "mather.h"

#define DEF_TIME_STEP 1
#define DEF_SIM_STEPS 100
#define DEF_STEP_PER_OUTPUT 1

using namespace std; 

// For debugging
void printF(Mass m){
	cout << "Forces:\tX = " << m.getCumalForces().x << "\tY = " << m.getCumalForces().y << "\tZ = " << m.getCumalForces().z << endl; 
} 
void printVel(Mass m){
	cout << "Velocities:\tX = " << m.getVelocity().x << "\tY = " << m.getVelocity().y << "\tZ = " << m.getVelocity().z << endl; 
} 
void printPos(Mass m){
	cout << "Pos:\tX = " << m.getPos().x << "   \tY = " << m.getPos().y << "   \tZ = " << m.getPos().z << endl; 
} 

int main(int argc, char ** argv)
{
	double TIME_STEP = DEF_TIME_STEP;
	unsigned int SIM_STEPS = DEF_SIM_STEPS;
	unsigned int STEP_PER_OUTPUT = DEF_STEP_PER_OUTPUT;  

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

	initConsts(); 

	vector<Mass> massVector; 

	Mass earth, moon; 
	earth.setMass(scientificNotation(5.9736, 24)); 
	earth.setName("Earth"); 
	moon.setMass(scientificNotation(7.349, 22)); 
	moon.setName("Moon"); 
	moon.setPos(scientificNotation(3.626, 8), 0, 0); 
	moon.setVelocity(0, 0, 1076); 

	massVector.push_back(earth);
	massVector.push_back(moon);
	

	cout << "G:\t" << G << endl; 
	cout << "Earth Mass:\t" << earth.getMass() << endl; 
	cout << "Moon Mass:\t" << moon.getMass() << endl; 
	
	//Mass massVector[10]; 
	
	/*
	// Init random masses
	for(int i=0; i<10; i++)
	{
		cartesian tempCart; 
		tempCart.x = rand()%1000; 
		tempCart.x = rand()%1000; 
		tempCart.z = rand()%1000; 
		massVector[i].setPos(tempCart); 
	}*/



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
				/*cout<<"Timestep: " << c << "\tObject Number: "<< i <<"\tObject Name: " << massVector[i].getName() 
				<< "\tPos: "<<massVector[i].getPos().x<<", "<<massVector[i].getPos().y
				<< ", "<<massVector[i].getPos().z<<endl; */

				//cout << massVector[1].getVelocity().z << endl; 
				//printF(massVector[1]);
				//printVel(massVector[1]);  
				printPos(massVector[1]); 
			}
			massVector[i].updateVelAndPos(TIME_STEP); 
			massVector[i].resetForces();  
		}	
		
		if(c > 100 && (abs(massVector[1].getPos().z) < 1))
		{
			cout << setprecision(51) << "Timestep: " << c << "\tObject Name: " << massVector[1].getName() << "\tPos: "<<massVector[1].getPos().x<<", "<<massVector[1].getPos().y<<", "<<massVector[1].getPos().z<<endl;
			printf("\n%f\n", massVector[1].getPos().z); 
			if((double)abs(massVector[1].getPos().z) < (double)1)
				cout << "Yes it's above" << endl; 
			int wait; 
			cin >> wait; 
		}

	}

	return 0;
}
