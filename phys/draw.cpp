#include "all_includes.h"


void drawAllShapes()
{
	// Parent orbitles 
	for(unsigned int i=0; i<numObjects; i++)
	{
		circles[i].setX(frameArr[(ext_frameCount+i)%(totalFrames*2)]/1000000000); 
		circles[i].setY(frameArr[(ext_frameCount+1+i)%(totalFrames*2)]/1000000000); 
		std::cout << "Frame: "<< ext_frameCount << "/" << totalFrames << "\t" << circles[i].getX() << "\t" << circles[i].getY() << std::endl;
		circles[i].draw(); 
		int temp; 
		//std::cin >> temp; 
	}

	Circle * center = new Circle(0, 0, 0, 0.005); 
	center->draw(); 
}


