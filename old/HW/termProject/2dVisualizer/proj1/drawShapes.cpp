#include "all_includes.h"

unsigned __int64 numObjects; 
unsigned __int64 framesPerObject; 
double maxDimension; 

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
		fread(&numObjects, 8, 1, vidFile);

		double tempDouble; 
		for(int i=0; i<numObjects; i++){
			fread(&tempDouble, 8, 1, vidFile);
			if(log10(tempDouble) > 29)
				//circles.push_back(*(new Circle(0, 0, 0, log10(tempDouble)/24)));
				circles.push_back(*(new Circle(0, 0, 0, 0.02)));
			else
				circles.push_back(*(new Circle(0, 0, 0, 0.001)));
		}

		fread(&framesPerObject, 8, 1, vidFile);
		frameArr = new double[framesPerObject*numObjects*2]; 
		for(int i=0; i<framesPerObject*numObjects*2; i++){
			fread(&tempDouble, 8, 1, vidFile);
			frameArr[i] = tempDouble; 
		}
		fclose(vidFile);
	}else{
		std::cout << "ERROR FILE NOT FOUND" << std::endl; 
	}
}

void initShapes()
{
	double avgDim = 0; 
	maxDimension = 0; 
	for(__int64 i=0; i<framesPerObject*numObjects*2; i++)
	{
		avgDim += fabs(frameArr[i]); 
		if(fabs(frameArr[i]) > maxDimension)
			maxDimension = fabs(frameArr[i]); 
	}
	avgDim /= (framesPerObject*numObjects*2); 
	maxDimension /= 1000; 
}


void drawAllShapes()
{
	for(unsigned int i=0; i<numObjects; i++)
	{
		circles[i].setX(frameArr[(ext_frameCount*numObjects*2+i*2)]/maxDimension); 
		circles[i].setY(frameArr[(ext_frameCount*numObjects*2+1+i*2)]/maxDimension); 
		if(i != 0)
			std::cout << "Frame: "<< ext_frameCount << "/" << framesPerObject << "\t" << circles[i].getX() << "     \t\t" << circles[i].getY() << std::endl;
		circles[i].draw(); 
		int temp; 
		//std::cin >> temp; 
	}

	//Circle * center = new Circle(0, 0, 0, 0.005); 
	//center->draw(); 
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
