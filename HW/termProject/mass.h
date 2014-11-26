typedef struct cartesian{
	double x; 
	double y;
	double z; 
}

class mass
{
	private:
		double mass;						// The mass of this mass in Kg
		cartesian position; 				// The cartesian position of this mass
		cartesian velocity; 				// Decomposed velocity of this mass
		cartesian cumalForces;		// The cumulative forces acting on this mass in Newtons
	public:
		// Getters and setters
		double getMass(); 
		cartesian getPos(); 
		cartesian getVelocity(); 
		void setMass(double newMass); 
		void setPos(double newPos); 
		void setVelocity(double newVelocity); 
		// Updates and stuff
		void resetForces(); 
		void addForce(cartesian force);
		cartesian updateVelAndPos(double timeStep);
}
