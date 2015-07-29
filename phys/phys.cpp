#include "phys.h"


void gravity(double uniMass, cart uniMassDist, std::vector<Shape> v)
{

}

bool collide(Shape s1, Shape s2)
{

}

void collideAndResolve(std::vector<Shape> v)
{

}

void resolveCollision(Shape s1, Shape s2, double dampingConst)
{

}

void updatePosWrap(cart worldLimits, std::vector<Shape> v)
{

}

void advanceSim(double t, std::vector<Shape> v)
{
	// Update position of all shapes

	// Detect and resolve all collisions

	// If worldwrap is on, worldwrap all objects
}
