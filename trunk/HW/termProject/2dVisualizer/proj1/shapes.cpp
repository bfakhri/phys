#include "all_includes.h"

int drawRectangle(float width, float height, bool filled)
{
	if((width < 0) || (height < 0))
	{
		return -1;	// Error, width/height are invalid
	}
	else
	{
		if(filled)
		{
			// Draw filled rectangle
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
		else
		{
			// Draw unfilled rectangle
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); 
		}
		glBegin(GL_POLYGON);
			glVertex2f(-width/2, height/2);		// Top Left
			glVertex2f(width/2, height/2);		// Top Right
			glVertex2f(width/2, -height/2);		// Bottom Right
			glVertex2f(-width/2, -height/2);	// Bottom Left
		glEnd(); 
		return 0;
	}
}

void drawCircle(float radius, unsigned int samples, bool filled)
{
	if(filled)
	{
		// Draw filled circle
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	else
	{
		// Draw unfilled circle
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); 
	}
	glPushMatrix();		// Ensures no corruption due to floating point precision
	glBegin(GL_POLYGON);
		for(unsigned int i=0; i<samples; i++)
		{
			glVertex2f(radius*cos(2*PI*i/samples), radius*sin(2*PI*i/samples)); 
		}
	glEnd();
	glPopMatrix();		// Restores original matrix
}

void drawPartialCircle(float radius, float startAngle, float endAngle, unsigned int samples)
{
	glPushMatrix();		// Ensures no corruption due to floating point precision

	float elapsedAngle = abs(endAngle - startAngle); 
	glBegin(GL_LINE_STRIP);
		for(unsigned int i=0; i<samples; i++)
		{
			glVertex2f(radius*cos((i*elapsedAngle/samples+startAngle)*PI/180), 
				radius*sin((i*elapsedAngle/samples+startAngle)*PI/180)); 
		}
	glEnd();
	glPopMatrix();		// Restores original matrix
}


void drawTriangle(float x1, float y1, float x2, float y2, float x3, float y3, bool filled)
{
		if(filled)
		{
			// Draw filled circle
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}
		else
		{
			// Draw unfilled circle
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); 
		}
		glBegin(GL_POLYGON);
			glVertex2f(x1, y1);		
			glVertex2f(x2, y2);	
			glVertex2f(x3, y3);		
		glEnd(); 
}

int drawRocket(float maxHeight, float maxWidth)
{
	int success = 0;	// -1 if failure
	
	// Calc temporary variables
	float coneHeight = 0.25*maxHeight; 
	float bodyHeight = 0.60*maxHeight; 
	float nozzleHeight = 0.15*maxHeight; 
	float bodyWidth = 0.50*maxWidth; 
	float finWidth = 0.50*maxWidth/2;	// Divided by 2 due to having two fins
	float finHeight = coneHeight;
	float nozzleWidth = 0.5*finWidth; 
	
	glPushMatrix();		// Save state of the matrix
	glPushMatrix();		// Save state of the matrix

	// Draw body and cone
	success = drawRectangle(bodyWidth, bodyHeight, false);						// Draw unfilled rectange for body
	glTranslatef(0.0, bodyHeight/2, 0.0);										// Goto top of rectangle
	drawTriangle(-bodyWidth/2, 0.0, bodyWidth/2, 0.0, 0.0, coneHeight, false);	// Draws cone
	
	glPopMatrix();		// Restore saved matrix 1

	// Draw fins/nozzle
	glTranslatef(0.0, -bodyHeight/2, 0.0);															// Goto bottom of rectangle
	drawTriangle(-(bodyWidth/2+finWidth), 0.0, -bodyWidth/2, 0.0, -bodyWidth/2, finHeight, false);	// Left fin
	drawTriangle((bodyWidth/2+finWidth), 0.0, bodyWidth/2, 0.0, bodyWidth/2, finHeight, false);		// Right fin
	drawTriangle(0.0, 0.0, -nozzleWidth, -nozzleHeight, nozzleWidth, -nozzleHeight, false);			// Nozzle

	glPopMatrix();		// Restore saved matrix 2

	return success; 
}

void drawCloud(float maxWidth, unsigned int samples)
{
	float startAng = 20; 
	float endAng = 160; 
	float elapsedAng = endAng - startAng; 
	float radius = maxWidth/(3*sin(elapsedAng*PI/180)/sin((180-elapsedAng)/2*PI/180)+2); 
	float loafWidth = radius*sin(elapsedAng*PI/180)/sin((180-elapsedAng)/2*PI/180); 
	unsigned int eachSample = samples/8; 
	float loafHeight = radius - radius*sin((180-elapsedAng)/2*PI/180); 
	
	// Save current matrix 3 times
	glPushMatrix();
	glPushMatrix(); 
	glPushMatrix(); 

	// Top loafs
	drawPartialCircle(radius, startAng, endAng, eachSample);	// middle loaf
	glTranslatef(-loafWidth, 0.0, 0.0);
	drawPartialCircle(radius, startAng, endAng, eachSample);	// Left loaf
	glTranslatef(2*loafWidth, 0.0, 0.0);
	drawPartialCircle(radius, startAng, endAng, eachSample);	// right loaf

	glPopMatrix(); 

	// Side loafs
	glTranslatef(-3*loafWidth/2, -loafHeight, 0.0); 
	drawPartialCircle(radius, 90, 270, eachSample);
	glTranslatef(2*3*loafWidth/2, 0.0, 0.0); 
	drawPartialCircle(radius, 270, 90, eachSample);
	
	glPopMatrix(); 
	glRotatef(180, 0.0, 0.0, 1.0); 
	glTranslatef(0.0, loafHeight*2, 0.0);

	drawPartialCircle(radius, startAng, endAng, eachSample);
	glTranslatef(-loafWidth, 0.0, 0.0);
	drawPartialCircle(radius, startAng, endAng, eachSample);
	glTranslatef(2*loafWidth, 0.0, 0.0);
	drawPartialCircle(radius, startAng, endAng, eachSample);

	glPopMatrix(); 

}
