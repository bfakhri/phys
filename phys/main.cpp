// Glut files
#include <GL/glut.h>
#include <GL/gl.h>

// I/O
#include <iostream>

// Physics
#include "phys.h"


// Shapes
#include "shape.h"
#include "sphere.h"
#include "cube.h"
#include "world.h"
#include "commonFunctions.h"

// Vector holding all worldly shapes
// May not include user-controlled ones
std::vector<Shape*> worldShapes;
	
cart org = {0, 0, -20};
cart max = {10, 10, 10};
cart min = {-10, -10, -10};

// Dummy main and function just for compiling purposes
void display()
{
	// Clear screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	// Green for shapes
	glColor3f(0.0, 1.0, 0.05);

	// Draw boundaries
	drawBoundaries(org, min, max);

	// Draws shapes in vector
	for(int i=0; i<worldShapes.size(); i++)	
		worldShapes[i]->draw(org);

	// Sends buffered commands to run
	glutSwapBuffers();

}

static void idle()
{
	advanceSim(1, worldShapes);
	enforceBoundaries(worldShapes, min, max);
	// Calls the display function 
	display();
}

int main(int argc, char **argv)
{
	// Just for testing
	std::cout<< G_CONST << std::endl;

	// Init shape vector
	
	cart tMaxPos = {10, 10, 30};
	cart tMaxVel = {0.1, 0.1, 0.1};
	for(int i=0; i<100; i++)
	{
		
		worldShapes.push_back((Sphere*)randomShape(0.1, 2, 9999999999999999999999.0, 99999999999999999999999.0, max, tMaxVel));
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






	glutMainLoop();
}
/*
// Launches a frame of the game
void display()
{
	// clear the screen 
	glClear(GL_COLOR_BUFFER_BIT);
	
	// Ensures green color
	glColor3f(0.0, 1.0, 0.05);

	// Holds shapes/breakout room
	glLoadIdentity(); 
	gluOrtho2D(-1.0, 1.0, -0.8, 0.8);
	glViewport(ext_winWidthOffset,						// Arithmetic to format viewport 
		0.2*ext_winHeight+ext_winHeightOffset, 			// to preserve original aspect ratios
		ext_winWidth-ext_winWidthOffset*2, 				// (squares remain square)
		0.8*(ext_winHeight-ext_winHeightOffset*2));		
														
	drawAllShapes();	// Draws breakout room
	glViewport(0, 0, ext_winWidth, ext_winHeight);		// Resets viewport
	
	// Holds the scoreboard/info panel
	glLoadIdentity(); 
	gluOrtho2D(-1.0, 1.0, -0.2, 0.2);
	glViewport(ext_winWidthOffset,						// Arithmetic to format viewport 
		ext_winHeightOffset, 							// to preserve original aspect ratios
		ext_winWidth-ext_winWidthOffset*2, 				// (squares remain square)
		0.2*(ext_winHeight-ext_winHeightOffset*2));		
														
	drawScore();		// Draws the score/timer
	glViewport(0, 0, ext_winWidth, ext_winHeight);		// Resets viewport

	// ready to draw now! forces buffered OGL commands to execute  
	glutSwapBuffers();
}

// Ensures that the breakout room and the scoreboard together are only as
// big as the smallest dimension. This ensures everything fits
void reshape(int newWidth, int newHeight)
{
	ext_winWidth = glutGet(GLUT_WINDOW_WIDTH);		// Gets window width
	ext_winHeight = glutGet(GLUT_WINDOW_HEIGHT);	// Gets window height
	if(ext_winWidth > ext_winHeight)
	{
		// Height is limiting 
		ext_winWidthOffset = (ext_winWidth - ext_winHeight)/2; 
		ext_winHeightOffset = 0; 
	}
	else
	{
		// Width is limiting 
		ext_winHeightOffset = (ext_winHeight - ext_winWidth)/2; 
		ext_winWidthOffset = 0; 
	}
}

// Detects when special keys on keyboard are pressed
void specialKeyPressed(int key, int x, int y)
{

}

// Detect non-special key presses
// Used for firing missile with space-bar
void keyPressed(unsigned char key, int x, int y)
{
	switch(key)
	{
	case ' ':			// Space-Bar
		//fireMissile(); 
		ext_score++; 
		break; 
	}
}

// Detect menu selections
void menuSelection(int option)
{
	switch(option)
	{
	case GAME_RESET:
		// reset game
		ext_secGameTime = 0; 
		ext_frameCount = 0; 
		ext_score = 0; 
		break;
	case GAME_QUIT:
		// quits game
		exit(0); 
		break;
	}
}
// Detect submenu selections
void submenuSelection(int option)
{
	switch(option)
	{
	case GAME_FILLED:
		// fills objects
		ext_filled = true; 
		break;
	case GAME_WIRE:
		// outlines objects
		ext_filled = false;
		break;
	}
	glutSwapBuffers(); 
}


// Initialize states -- called before  
void init()
{
	// Init Globals
	ext_filled = false;	// Objects are filled
	ext_refreshRate = 30; 
	ext_clocksPerRefresh = CLOCKS_PER_SEC/ext_refreshRate;
	ext_lastRefreshClock = 0; 
	ext_winWidthOffset = 0;
	ext_winHeightOffset = 0; 
	ext_secGameTime = 0;
	ext_frameCount = 0; 
	ext_missileSpeed = 0.03;
	ext_wallUniTargets = 15;
	ext_leftUniTargets = 4; 
	ext_rightUniTargets = 4; 

	// set background clear color to black  
	glClearColor(0.0, 0.0, 0.0, 0.0);
	// set current color to white  
	glColor3f(1.0, 1.0, 1.0);

	// identify the projection matrix that we would like to alter  
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	// the window will correspond to these world coorinates  
	gluOrtho2D(-1.0, 1.0, -1.0, 1.0);

	// identify the modeling and viewing matrix that can be modified from here on  
	// we leave the routine in this mode in case we want to move the object around  
	// or specify the camera  
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// For playback
	initFrames();
	// Initializes the playing field
	initShapes(); 
	// Initializes scoreboard
	initScore(); 
}

// Called when not doing anything
// Here we use it to time frames
static void idle()
{
	// Check if enough clocks have elapsed
	if((clock() - ext_lastRefreshClock) >= ext_clocksPerRefresh)
	{ 
		ext_lastRefreshClock = clock();			// Records last time a frame was launched
		ext_frameCount+=2;						// Increments frame counter
		if(ext_frameCount >= ext_refreshRate)	// Checks to see if a second has gone by
		{
			ext_secGameTime++;		// Increments game time
			//ext_frameCount = 0;		// Resets frame count
		}
		display();		// Launches the new frame
	}
}


// Make the menu  
int setupMenu (void)
{
	int polygonMen = glutCreateMenu(submenuSelection); 
	glutAddMenuEntry("WIRE", GAME_WIRE); 
	glutAddMenuEntry("FILLED", GAME_FILLED); 

	int mainMen = glutCreateMenu (menuSelection);
	glutAddMenuEntry ("RESET", GAME_RESET);
	glutAddMenuEntry ("QUIT", GAME_QUIT);
	glutAddSubMenu("POLYGON", polygonMen); 

	return mainMen; 
}


// The main program  

int main(int argc, char** argv)
{
	// create a window, position it, and name it  
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(200, 200);
	glutCreateWindow("Project One: Breakout");

	// create a callback routine for (re-)display  
	glutDisplayFunc(display);

	// Function called when idle  
	glutIdleFunc(idle); 
	
	// Function called when window is resized  
	glutReshapeFunc(reshape); 

	// Function called when special key is pressed  
	glutSpecialFunc(specialKeyPressed); 

	// Function called when non-special key is pressed   
	glutKeyboardFunc(keyPressed); 

	// Menu Setup
	setupMenu();
	glutAttachMenu(GLUT_RIGHT_BUTTON);

	// init some states  
	init();

	// entering the event loop  
	glutMainLoop();

	// code here will not be executed  
}
*/
