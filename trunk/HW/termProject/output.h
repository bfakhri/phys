#ifndef OUTPUT_H
#define OUTPUT_H

#include <stdio.h>
#include <stdint.h>
#include <vector> 
#include "mass.h"

extern FILE * ext_vidFile;

void initOutput();
void initOutput(char * name);
void outputObjectCount(uint32_t objectCount);
void outputObjectMass(double mass);
void outputSimSteps(uint32_t simSteps);
void outputFrames(double x, double y); 
void outputClose(); 
void outputResults(vector<Mass> massVector); 

#endif 
