// Main

#include "mass.h"
#include <iostream>
#include <stdlib.h>

#define TIME_STEP 0.01
#define SIM_STEPS 100
using namespace std; 

int main(int argc, char ** argv)
{
	Mass earth;
	earth.setMass(5.9736*100); 

	Mass massArr[10]; 

	for(int i=0; i<10; i++)
	{
		cartesian tempCart; 
		tempCart.x = rand()%1000; 
		tempCart.x = rand()%1000; 
		tempCart.z = rand()%1000; 
		massArr[i].setPos(tempCart); 
	}

	// Simulation loop
	for(int c=0; c<SIM_STEPS; c++)
	{
		// Influences
		for(int i=0; i<10; i++)
		{
			massArr[i].resetForces(); 
			for(int j=0; j<10; j++)
			{
				if(i != j)
				{
					massArr[i].influence(massArr[j]); 
				}
			}

		}
		
		// Update position
		for(int i=0; i<10; i++)
		{
			cout<<"Object: \t"<<i<<"\tPos: "<<massArr[i].getPos().x<<", "<<massArr[i].getPos().y<<", "<<massArr[i].getPos().z<<endl;
			massArr[i].updateVelAndPos(TIME_STEP);  
		}	
	}

	return 0;
}
