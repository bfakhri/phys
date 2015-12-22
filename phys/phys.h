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
#include <thread>	// For multithreading
#include <omp.h>	// For multithreading
#include <chrono>	// For timing info
#include <iostream>	// For Debugging


// Simulation parameters
// Default period for each step (seconds)
const double SIM_FPS = 200;						// FPS of the physics engine
const double SIM_T = 0.005;						// Default sim period
const double G_CONST = 0.0000000000667384;		// Gravitational constant G
const double GRAV_ACCEL = 9.81;					// Accel due to gravity in m/s^2
const double SPRING_CONST = 20;					// Need to tweak this
const double BOUNCE_COEFF = 1.0;				// Determines how much energy is given back after a bounce
const double DRAG_COEFF = 0.06;					// Drag coefficient * density of air
// World stuff
// Directions
const cart DIR_UP =	{ 0,  1,  0};
const cart DIR_DOWN =	{ 0, -1,  0};
const cart DIR_RIGHT =	{ 1,  0,  0};
const cart DIR_LEFT =	{-1,  0,  0};
const cart DIR_FWRD =	{ 0,  0, -1};
const cart DIR_BACK =	{ 0,  0,  1};
const cart PHYS_ORIG = {0, 0, -2.5};			// Origin of sim relative to drawing coords
const cart physBoundaryMax = {1, 1, 1};		// Maximum coordinates of physics sim
const cart physBoundaryMin = {-1, -1, -1};	// Minimum coordinates of physics sim

///////////////////
// Helper Functions 
///////////////////

// Returns the negative of the vector
cart negate(cart c);

// Returns length of vector
double length(cart c);

// Returns dot product of the vectors
double dotProd(cart c1, cart c2);

// Multiplies vectors by components, returning a vector of products
cart multComponents(cart c1, cart c2);

// Returns a vector of quotients 
cart divComponents(cart dividend, cart divisor);

// Returns distance between two cartesian coordinates
double distance(cart c1, cart c2);

// Return distance between two shapes
double distance(Shape* s1, Shape* s2);

// Resets the force vectors of an object 
// (both rotational and translational)
void resetForces(std::vector<Shape*> v);



////////////////////
// Force Functions		-- Forces like gravity, friction, etc
////////////////////

// Imparts force of air friction on all shapes in vector
void airFoceAll(std::vector<Shape*> v);

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
void gravAllMass(cart uniMassDist, std::vector<Shape*> v);



//////////////////////
// Collision functions
//////////////////////

// Determines whether two shapes are moving towards each other
// Usefull to see if we need to resolve a collision
bool movingTowards(Shape* s1, Shape* s2);

// Detects whether two shapes are colliding or not
// for any shape combinations
bool collide(Shape* s1, Shape* s2);

// Detects collisions and resolves them for all objects
void collideAndResolve(std::vector<Shape*> v);

// Resolves a collision
// - Includes option for damping/friction
void resolveCollision(Shape* s1, Shape* s2, double dampingConst);

// Resolve a spring-like collision
// Shapes are treated like rubber/spring masses
void resolveCollisionSpring(Shape* s1, Shape* s2);

// Bounces shape off of a wall
void bounce(Shape* s, cart wall);

///////////////////////
// Simulation Functions 
///////////////////////

// Updates translational velocities according to the acceleration
void t_updateVel(double t, std::vector<Shape*> v);

// Updates rotational velocities according to rotational accelerations
void r_updateVel(double t, std::vector<Shape*> v);


// Move one timestep using the translational forces on all the objects
void t_updatePos(double t, std::vector<Shape*> v);

// Move one timestep using the rotational forces (torques)  on all the objects
void r_updatePos(double t, std::vector<Shape*> v);

// Move one timestep both translational and rotational positions 
void updateVelPosAndReset(double t, std::vector<Shape*> v);

// Updates positions of shapes to simulate the world wrapping around the edges
// Like in pacman where if you leave the world on the right extreme you appear on the left extreme
// MAKE SURE THIS IS MATHEMATICALLY SOUND
void wrapWorld(cart worldLimits, std::vector<Shape*> v);

// Advances vector by one time step of length t
void advanceSim(double t, std::vector<Shape*> v);

// Makes shapes bounce off of the walls of the sim-world
void enforceBoundaries(std::vector<Shape*> s, cart min, cart max, double dampingConst);

// The function that will have its own thread to run the
// simulation parallel to the rendering engine
void physicsThread(std::vector<Shape*> v);
#endif
