// NEED BIG NUM BOOST LIBRARY IN ORDER TO EXPRESS SOME OF THESE NUMBERS
#ifndef MASS_H
#define MASS_H

#include <math.h>
#include <string> 

using namespace std; 

extern double G;

typedef struct{
	double x; 
	double y;
	double z; 
}cartesian;

void initConsts(); 

class Mass
{
	private:
		string name; 			// Name of the object, if applicable
		double objectMass;		// The Mass of this Mass in Kg
		cartesian position;		// The cartesian position of this Mass
		cartesian velocity;		// Decomposed velocity of this Mass
		cartesian cumalForces;		// The cumulative forces acting on this Mass in Newtons
	public:
		// Getters and setters
		Mass();
		string getName(); 
		double getMass(); 
		cartesian getPos(); 
		cartesian getVelocity(); 
		cartesian getCumalForces();
		void setName(string newName); 
		void setMass(double newMass); 
		void setPos(cartesian newPos); 
		void setPos(double x, double y, double z); 
		void setVelocity(cartesian newVelocity); 
		void setVelocity(double x, double y, double z); 

		// Updates and stuff
		void resetForces(); 
		void addForce(cartesian force);
		void addForce(double x, double y, double z);
		double newtonGrav(double objMass, double distance); 
		void influence(Mass obj); 
		cartesian updateVelAndPos(double timeStep);
};

#endif
