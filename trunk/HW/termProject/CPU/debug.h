#ifndef DEBUG_H
#define DEBUG_H

#include <iostream>
#include <stdint.h>
#include "mass.h" 

// For debugging
void printTime(uint64_t seconds);
void printF(Mass m);
void printVel(Mass m);
void printPos(Mass m);
void printDist(Mass m1, Mass m2);

#endif
