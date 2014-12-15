// This file MIGHT need error checking capabilities for robustness
#include "output.h"

FILE * ext_vidFile; 

void initOutput(){
	ext_vidFile = fopen("vidFile.bin", "wb"); 
}

void outputObjectCount(uint32_t objectCount){
	fwrite ( (const void*)(&objectCount), 4, 1, ext_vidFile );
}

void outputObjectMass(double mass){
	fwrite ( (const void*)(&mass), 8, 1, ext_vidFile );
}

void outputSimSteps(uint32_t simSteps){
	fwrite ( (const void*)(&simSteps), 4, 1, ext_vidFile );
}

void outputFrames(double x, double y){
	fwrite ( (const void*)(&x), 8, 1, ext_vidFile );
	fwrite ( (const void*)(&y), 8, 1, ext_vidFile );
}

void outputClose(){
	fclose(ext_vidFile); 
}
