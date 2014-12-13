// NEED BIG NUM BOOST LIBRARY IN ORDER TO EXPRESS SOME OF THESE NUMBERS


#include <math.h>

#define G (6.67384*0.00000000001)

typedef struct{
	double x; 
	double y;
	double z; 
}cartesian;


class Mass
{
	private:
		double objectMass;		// The Mass of this Mass in Kg
		cartesian position;		// The cartesian position of this Mass
		cartesian velocity;		// Decomposed velocity of this Mass
		cartesian cumalForces;		// The cumulative forces acting on this Mass in Newtons
	public:
		// Getters and setters
		Mass();
		double getMass(); 
		cartesian getPos(); 
		cartesian getVelocity(); 
		void setMass(double newMass); 
		void setPos(cartesian newPos); 
		void setVelocity(cartesian newVelocity); 
		// Updates and stuff
		void resetForces(); 
		void addForce(cartesian force);
		double newtonGrav(double objMass, double distance); 
		void influence(Mass obj); 
		cartesian updateVelAndPos(double timeStep);
};
