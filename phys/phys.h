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
#include <math.h>
#include <vector>

// Gravitational Constant
const double G_CONST = 0.0000000000667384;

// Returns distance between two cartesian coordinates
double distance(cart c1, cart c2);

// Return distance between two shapes
double distance(Shape* s1, Shape* s2);

// Returns the force of gravity between two masses seperated by a distance
double gravForce(double m1, double m2, double dist);


// Influence by gravity - mass1 will influence mass2 by adding a force
// to its force vector
void gravInfluenceShape(Shape m1, Shape m2);

// Influence shape by mass at a predefined distance
void gravInfluenceMass(double uniMass, cart uniMassDist, Shape* s);

// Adds gravitational forces acting on all object by all objects
// Include option for universal gravity source 
// - uniMass is the mass of the universal gravity source
// - uniMassDistance is the distance of all object to that mass from each component (same for all objects)
void gravity(double uniMass, cart uniMassDist, std::vector<Shape*> v);

// Detects whether two shapes are colliding or not
// for any shape combinations
bool collide(Shape* s1, Shape* s2);

// Detects whether two Spheres are colliding or not
bool collide(Sphere* s1, Sphere* s2);

// Detects collisions and resolves them for all objects
void collideAndResolve(std::vector<Shape*> v);

// Detects collisions and resolves them for spheres
void collideAndResolve(std::vector<Sphere*> v);

// Resolves a collision
// - Includes option for damping/friction
void resolveCollision(Shape s1, Shape s2, double dampingConst);

// Updates positions of shapes to simulate the world wrapping around the edges
// Like in pacman where if you leave the world on the right extreme you appear on the left extreme
// MAKE SURE THIS IS MATHEMATICALLY SOUND
void updatePosWrap(cart worldLimits, std::vector<Shape*> v);

// Advances whole simulation by one time step of length t
void advanceSim(double t, std::vector<Shape*> v);


#endif
