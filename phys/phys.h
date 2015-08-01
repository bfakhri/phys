// This file defines the physics/math related functions of the project
// The following physical phenomena shall be defined here
// - Interparticle forces such as gravity
// - Collision detection/resolution
// - Frictional/damping effects of collisions and interactions

// This file should also provide functionality for toggling phenomena on and off

#ifndef PHYS_H
#define PHYS_H

#include "mather.h"
#include "shape.h"
#include "sphere.h"
#include <math.h>
#include <vector>

// Gravitational Constant
const double G_CONST = 0.0000000000667384;


///////////////////
// Helper Functions 
///////////////////

// Returns distance between two cartesian coordinates
double distance(cart c1, cart c2);

// Return distance between two shapes
double distance(Shape* s1, Shape* s2);

// Resets the force vectors of an object 
// (both rotational and translational)
void resetForces(std::vector<Shape*> v);



////////////////////
// Gravity Functions 
////////////////////

// Returns the force of gravity between two masses seperated by a distance
double gravForce(double m1, double m2, double dist);

// Influence by gravity - mass1 will influence mass2 by adding a force
// to its force vector
void gravPull(Shape* m1, Shape* m2);

// Influence shape by mass at a predefined distance
void gravPull(double uniMass, cart uniMassDist, Shape* s);

// Adds gravitational forces acting on all object by all objects
void gravAllShapes(std::vector<Shape*> v);

// Adds grav force to all object in vector by a universal mass/gravity source
// - uniMass is the mass of the universal gravity source
// - uniMassDistance is the distance of all object to that mass from each component 
//	(same for all objects)
void gravAllMass(double uniMass, cart uniMassDist, std::vector<Shape*> v);



//////////////////////
// Collision functions
//////////////////////

// Detects whether two shapes are colliding or not
// for any shape combinations
bool collide(Shape* s1, Shape* s2);

// Detects collisions and resolves them for all objects
void collideAndResolve(std::vector<Shape*> v);

// Resolves a collision
// - Includes option for damping/friction
void resolveCollision(Shape* s1, Shape* s2, double dampingConst);



///////////////////////
// Simulation Functions 
///////////////////////

// Move one timestep using the translational forces on all the objects
void t_advancePos(double t, std::vector<Shape*> v);

// Move one timestep using the rotational forces (torques)  on all the objects
void r_advancePos(double t, std::vector<Shape*> v);

// Move one timestep both translational and rotational positions 
void advancePosAndReset(double t, std::vector<Shape*> v);

// Updates positions of shapes to simulate the world wrapping around the edges
// Like in pacman where if you leave the world on the right extreme you appear on the left extreme
// MAKE SURE THIS IS MATHEMATICALLY SOUND
void wrapWorld(cart worldLimits, std::vector<Shape*> v);

// Advances vector by one time step of length t
void advanceSim(double t, std::vector<Shape*> v);


#endif
