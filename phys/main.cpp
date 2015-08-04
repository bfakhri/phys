// Glut files
#include <GL/glut.h>
#include <GL/gl.h>

// I/O
#include <iostream>
#include <stdlib.h>	// For atoi()

// Multithreading
#include <thread>
#include <omp.h> 

// Timing
#include <chrono>

// Physics
#include "phys.h"

// Shapes
#include "shape.h"
#include "sphere.h"
#include "cube.h"

const double DRAW_FPS = 60;

// Vector holding all worldly shapes
// May not include user-controlled ones
std::vector<Shape*> worldShapes;

// Timing variables	
using namespace std::chrono;
high_resolution_clock::time_point last = high_resolution_clock::now();
high_resolution_clock::time_point now = high_resolution_clock::now();


void display()
{
	// Clear screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	// Draw boundaries
	drawBoundaries(physOrigin, physBoundaryMin, physBoundaryMax);

	// Draws shapes in vector
	for(int i=0; i<worldShapes.size(); i++)	
		worldShapes[i]->draw(physOrigin);

	//std::cout << worldShapes[0]->t_position.x << std::endl;

	// Sends buffered commands to run
	glutSwapBuffers();

}

void idle()
{
	high_resolution_clock::time_point now = high_resolution_clock::now();
	//std::cout << duration_cast<std::chrono::milliseconds>(now - last).count() << "   " << 1000/DRAW_FPS <<  std::endl;
	if(duration_cast<std::chrono::milliseconds>(now - last).count() >= 1000/DRAW_FPS)
	{
		display();
		last = high_resolution_clock::now();
	}
}


void initOGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(1000, 1000);
	glutInitWindowPosition(200, 200);
	glutCreateWindow("Phys");
	glutDisplayFunc(display);

	// Function called when idle  
	glutIdleFunc(idle); 

	// Enable CULLING
	// Ensures faces of objects facing away from the camera are not rendered
	// Back facing faces will not obscure the front faces consequently
	glEnable(GL_CULL_FACE);

	// set background clear color to black 
	glClearColor(0.0, 0.0, 0.0, 0.0);

	// For Depth 
	glEnable(GL_DEPTH_TEST);

	// For Antialiasing - test this 
	//glEnable(GL_MULTISAMPLE);

	// identify the projection matrix that we would like to alter 
	glMatrixMode(GL_PROJECTION);

	// Set up perspective projection 
	gluPerspective(90, 1.0, 0.01, 120.0); 

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void initSim(int numShapes)
{
	// Init shape vector	
	cart tMaxVel = {8.8, 5.8, 3.8};
	for(int i=0; i<numShapes; i++)
	{
		
		worldShapes.push_back((Sphere*)randomShape(0.1, 2, 90000000000.0, 9000000000000.0, physBoundaryMax, tMaxVel));
		worldShapes[i]->r_velocity.x = i*3.14/100;
		worldShapes[i]->r_velocity.y = i*3.14/100;
		worldShapes[i]->r_velocity.z = i*3.14/100;
		
	}
	cart pos = {0, 0, -2};
	cart rot = {01, 01, 01};
	cart zer = {0, 0, 0};
	worldShapes.push_back(new Cube(2, 1, pos, zer, zer, rot));

	// Timing variables
	
}


int main(int argc, char **argv)
{
	// Parameters into 
	int numObjects;
	if(argc == 2)
		numObjects = atoi(argv[1]);
	else
		numObjects = 10;
		
	// Init OpenGL constructs
	initOGL(argc, argv);	

	// Init world objects
	initSim(numObjects);

	// Start the physics engine:
	std::thread peThread(physicsThread, worldShapes);

	// Enter infinite loop - does not return
	glutMainLoop();
	
	// I don't think this is necessary
	peThread.join();
}
