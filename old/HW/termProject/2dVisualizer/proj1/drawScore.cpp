#include "all_includes.h"

// Text Strings and sizes
std::string scoreStr = "SCORE: ";
std::string timeStr = "TIME:  "; 
std::string targetStr= "Targets Left:"; 

void initScore()
{
	ext_score = 0; 
	ext_secGameTime = 0; 
	ext_targetsLeft = ext_wallUniTargets +	
		ext_leftUniTargets +  
		ext_rightUniTargets + 4;		// Add 4 for the center targets
}

void drawScore()
{ 
	// Scale so text is not HUGE
	glScaled(1.0f/SCALE_FACTOR, 1.0f/SCALE_FACTOR, 1.0f/SCALE_FACTOR); 
	
	// Draw Score
	glPushMatrix(); 
	glPushMatrix(); 
	glPushMatrix(); 
	
	glTranslatef(-TEXT_WIDTH*25, 0, 0);		// Scoots text to left
	for(unsigned int i=0; i<scoreStr.size(); i++)
	{
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, scoreStr[i]);	// Draws character
		glTranslatef(TEXT_WIDTH, 0, 0);								// Moves for next character
	}
	// Creates string from score
	std::string score = std::to_string((unsigned long long)ext_score); 
	for(unsigned int i=0; i<score.size(); i++)
	{
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, score[i]); 
		glTranslatef(TEXT_WIDTH, 0, 0); 
	}
	glPopMatrix(); 

	// Draw TIME
	glTranslatef(-TEXT_WIDTH*25, -2*TEXT_WIDTH, 0);		// Scoots text to left
	for(unsigned int i=0; i<timeStr.size(); i++)
	{
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, timeStr[i]); 
		glTranslatef(TEXT_WIDTH, 0, 0); 
	}
	// Creates string from time
	std::string time = std::to_string((unsigned long long)ext_secGameTime); 
	for(unsigned int i=0; i<time.size(); i++)
	{
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, time[i]); 
		glTranslatef(TEXT_WIDTH, 0, 0); 
	}
	glPopMatrix(); 

	// Draw Targets Left 
	glTranslatef(-TEXT_WIDTH*25, -4*TEXT_WIDTH, 0);		// Scoots text to left
	for(unsigned int i=0; i<targetStr.size(); i++)
	{
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, targetStr[i]); 
		glTranslatef(TEXT_WIDTH, 0, 0); 
	}
	// Creates string from time
	std::string targets = std::to_string((unsigned long long)ext_targetsLeft); 
	for(unsigned int i=0; i<targets.size(); i++)
	{
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, targets[i]); 
		glTranslatef(TEXT_WIDTH, 0, 0); 
	}
	glPopMatrix(); 
	


}
