typedef struct{
	double x; 
	double y;
	double z; 
}cartesian;

class mass
{
	private:
		double objectMass;		// The mass of this mass in Kg
		cartesian position;		// The cartesian position of this mass
		cartesian velocity;		// Decomposed velocity of this mass
		cartesian cumalForces;		// The cumulative forces acting on this mass in Newtons
	public:
		// Getters and setters
		mass();
		double getMass(); 
		cartesian getPos(); 
		cartesian getVelocity(); 
		void setMass(double newMass); 
		void setPos(cartesian newPos); 
		void setVelocity(cartesian newVelocity); 
		// Updates and stuff
		void resetForces(); 
		void addForce(cartesian force);
		cartesian updateVelAndPos(double timeStep);
};
