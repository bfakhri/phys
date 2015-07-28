// This file defines the physics related functions of the project
// The following physical phenomena shall be defined here
// - Interparticle forces such as gravity
// - Collision detection/resolution
// - Frictional/damping effects of collisions and interactions

// This file should also provide functionality for toggling phenomena on and off

#ifndef PHYS_H
#define PHYS_H

#include "shape.h"

// Adds gravitational forces acting on all object by all objects
// Include option for universal gravity source 
// - uniMass is the mass of the universal gravity source
// - uniMassDistance is the distance of all object to that mass from each component (same for all objects)
void gravity(double uniMass, cart uniMassDist);

// Detects whether two shapes are colliding or not
bool collide(Shape s1, Shape s2);

// Detects collisions and resolves them for all objects
void collideAndResolve();

// Resolves a collision
// - Includes option for damping/friction
void resolveCollision(Shape s1, Shape s2, double dampingConst);

// Updates positions of shapes to simulate the world wrapping around the edges
// Like in pacman where if you leave the world on the right extreme you appear on the left extreme
// MAKE SURE THIS IS MATHEMATICALLY SOUND
void updatePosWrap(cart worldLimits);

// Advances whole simulation by one time step of length t
void advanceSim(double t);


#endif
