// Glut files
#include <GL/glut.h>
#include <GL/gl.h>

// I/O
#include <iostream>

// Multithreading
#include <thread>

// Physics
#include "phys.h"


// Shapes
#include "shape.h"
#include "sphere.h"
#include "cube.h"

// Vector holding all worldly shapes
// May not include user-controlled ones
std::vector<Shape*> worldShapes;
	

// Dummy main and function just for compiling purposes
void display()
{
	// Clear screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	// Draw boundaries
	drawBoundaries(physOrigin, physBoundaryMin, physBoundaryMax);

	// Draws shapes in vector
	for(int i=0; i<worldShapes.size(); i++)	
		worldShapes[i]->draw(physOrigin);

	// Sends buffered commands to run
	glutSwapBuffers();

}

static void idle()
{
	// if(time > time...asdfa;sldkfj)
	display();
}

int main(int argc, char **argv)
{
	// Init shape vector	
	cart tMaxVel = {0.8, 0.8, 0.8};
	for(int i=0; i<4; i++)
	{
		
		worldShapes.push_back((Sphere*)randomShape(0.1, 2, 9999999999999999999999.0, 99999999999999999999999.0, physBoundaryMax, tMaxVel));
		worldShapes[i]->r_velocity.x = i*3.14/100;
		worldShapes[i]->r_velocity.y = i*3.14/100;
		worldShapes[i]->r_velocity.z = i*3.14/100;
		
	}
	cart pos = {0, 0, -2};
	cart rot = {01, 01, 01};
	cart zer = {0, 0, 0};
	worldShapes.push_back(new Cube(2, 1, pos, zer, zer, rot));
	
	glutInit(&argc, argv);
	
	// From original
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(1000, 1000);
	//glutInitWindowSize(200, 200);
	glutInitWindowPosition(200, 200);
	glutCreateWindow("Phys");
	glutDisplayFunc(display);
	// End from original

	// Function called when idle  
	glutIdleFunc(idle); 

	// Enable CULLING
	// Ensures faces of objects facing away from the camera are not rendered
	// Back facing faces will not obscure the front faces consequently
	glEnable(GL_CULL_FACE);

	// set background clear color to black 
	glClearColor(0.0, 0.0, 0.0, 0.0);
	// set current color to white 

	// For Depth 
	glEnable(GL_DEPTH_TEST);
	// More Depth
	//glDepthFunc(GL_LESS);

	// For Lighting 
	//glEnable(GL_LIGHTING);	
	//glEnable(GL_LIGHT0);
	//float ambientSettings[4] = {0.0, 0.7, 0.2, 1}; 
	//glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambientSettings); 

	// identify the projection matrix that we would like to alter 
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();   
	// Set up perspective projection 
	gluPerspective(90, 1.0, 0.01, 120.0); 
	//gluLookAt(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);


	// identify the modeling and viewing matrix that can be modified from here on 
	// we leave the routine in this mode in case we want to move the object around 
	// or specify the camera 
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	// Start the physics engine:
	std::thread peThread(physicsThread, worldShapes);


	glutMainLoop();
}
