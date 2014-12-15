#include "all_includes.h"

unsigned __int32 numObjects; 
unsigned __int32 totalFrames; 

/* Targets */
// Wall
std::vector<Circle> circles; 
double* frameArr; 

void initFrames()
{
	FILE * vidFile;
	vidFile = fopen ("vidFile.bin","rb");
	if (vidFile!=NULL)
	{
		fread(&numObjects, 4, 1, vidFile);
		fread(&totalFrames, 4, 1, vidFile);
		frameArr = new double[totalFrames*2]; 
		double bs; 
		//fread(&bs, 4, 1, vidFile);
		double tempDouble; 
		for(int i=0; i<totalFrames; i+=2){
			fread(&tempDouble, 8, 1, vidFile);
			frameArr[i] = tempDouble; 
			fread(&tempDouble, 8, 1, vidFile);
			frameArr[i+1] = tempDouble; 
		}
		fclose(vidFile);
	}else{
		std::cout << "ERROR FILE NOT FOUND" << std::endl; 
	}
}

void initShapes()
{
	for(int i=0; i<numObjects; i++){
		circles.push_back(*(new Circle(0, 0, 0, 0.02)));
	}
}


void drawAllShapes()
{
	// Parent orbitles 
	for(unsigned int i=0; i<numObjects; i++)
	{
		circles[i].setX(frameArr[(ext_frameCount+i*2)%(totalFrames*2)]/1000000000); 
		circles[i].setY(frameArr[(ext_frameCount+1+i*2)%(totalFrames*2)]/1000000000); 
		std::cout << "Frame: "<< ext_frameCount << "/" << totalFrames << "\t" << circles[i].getX() << "\t" << circles[i].getY() << std::endl;
		circles[i].draw(); 
		int temp; 
		//std::cin >> temp; 
	}

	Circle * center = new Circle(0, 0, 0, 0.005); 
	center->draw(); 
}


void drawDecorativeShapes()
{
	glLoadIdentity(); 
	glPushMatrix();
	glPushMatrix();
	glPushMatrix();
	glPushMatrix();
	glTranslatef(-0.75, 0.7, 0);
	drawCloud(0.3, DEF_SAMPLES*50); 
	glPopMatrix(); 
	glTranslatef(0.75, 0.7, 0);
	drawCloud(0.3, DEF_SAMPLES*50);
	glPopMatrix(); 
	glTranslatef(-0.75, -0.6, 0); 
	drawRocket(0.2, 0.1); 
	glPopMatrix(); 
	glTranslatef(0.75, -0.6, 0);
	drawRocket(0.2, 0.1); 
}
