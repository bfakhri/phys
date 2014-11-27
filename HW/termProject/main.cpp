// Main

#include "mass.h"

using namespace std; 

int main()
{
	mass huge;
	huge.setMass(10000000); 

	mass massArr[1000]; 

	for(int i=0; i<1000; i++)
	{
		cartesian tempCart; 
		tempCart.x = rand()%1000; 
		tempCart.x = rand()%1000; 
		tempCart.z = rand()%1000; 
		massArr[i].setPos(tempCart); 
	}

	// Simulation loop
	for(int i=0; i<1000; i++)
	{
		for(int i=0; i<1000; i++)
		{
			
		}
	}
