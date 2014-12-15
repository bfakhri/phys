#ifndef OUTPUT_H
#define OUTPUT_H

#include <stdio.h>

extern FILE * ext_vidFile;

void initOutput();
void outputObjectCount(uint32_t objectCount);
void outputObjectMass(uint32_t mass);
void outputSimSteps(uint32_t simSteps);
void outputFrames(double x, double y); 
void outputClose(); 
 
