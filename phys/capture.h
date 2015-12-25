// This file defines functions for capturing and outputting
// frames of the simulation. This will allow the simulation
// to be done in non-realtime but later viewed in realtime. 

#ifndef CAPTURE_H
#define CAPTURE_H

// Glut files
#include <GL/glut.h>
#include <GL/gl.h>

// I/O
#include <iostream>
#include <stdlib.h>	// For atoi()

// PNG Writing Library - libPNG
#include <png.h>

// Globals
#include "globals.h"

const uint32_t VIDEO_FPS = 30;

// Maybe make this a singleton in the future? 
class Recorder
{
private: 
	uint32_t frameCount;
	double time;
public:
	Recorder();
	void capture(float);	// Captures OGL frame to disk as PNG
};

int writeImage(char* filename, int width, int height, uint8_t** buffer, char* title);
#endif


