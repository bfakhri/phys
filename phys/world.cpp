#include "world.h"



void drawBoundaries(cart origin, cart min, cart max)
{
	glPushMatrix(); 
	// Go to physics origin
	glTranslatef(origin.x, origin.y, origin.z); 
	
	glBegin(GL_QUADS);
	
	//	Back	
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, max.y, min.z); 
	glVertex3f(min.x, max.y, min.z); 
	glVertex3f(min.x, min.y, min.z); 
	glVertex3f(max.x, min.y, min.z); 
	//	Right
	glColor3f(0.7, 0.7, 0.7);
	glVertex3f(max.x, max.y, max.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, max.y, min.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, min.y, min.z); 
	glColor3f(0.7, 0.7, 0.7);
	glVertex3f(max.x, min.y, max.z); 
	//	Top
	glColor3f(0.7, 0.7, 0.7);
	glVertex3f(max.x, max.y, max.z); 
	glColor3f(0.7, 0.7, 0.7);
	glVertex3f(min.x, max.y, max.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, max.y, min.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, max.y, min.z); 
	//	Left
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, max.y, min.z); 
	glColor3f(0.7, 0.7, 0.7);
	glVertex3f(min.x, max.y, max.z); 
	glColor3f(0.7, 0.7, 0.7);
	glVertex3f(min.x, min.y, max.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, min.y, min.z); 
	//	Bottom
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(max.x, min.y, min.z); 
	glColor3f(0.0, 0.0, 0.0);
	glVertex3f(min.x, min.y, min.z); 
	glColor3f(0.7, 0.7, 0.7);
	glVertex3f(min.x, min.y, max.z); 
	glColor3f(0.7, 0.7, 0.7);
	glVertex3f(max.x, min.y, max.z); 

	glEnd();

	glPopMatrix();
}


void enforceBoundaries(std::vector<Shape*> s, cart min, cart max)
{
	for(int i=0; i<s.size(); i++)
	{
		// One for each boundary 
		if((s[i]->t_position.x + s[i]->boundingSphere()) > max.x)
			s[i]->t_velocity.x *= -1.0;
		if((s[i]->t_position.y + s[i]->boundingSphere()) > max.x)
			s[i]->t_velocity.y *= -1.0;
		if((s[i]->t_position.z - s[i]->boundingSphere()) < max.x)
			s[i]->t_velocity.z *= -1.0;
		if((s[i]->t_position.x + s[i]->boundingSphere()) < min.x)
			s[i]->t_velocity.x *= -1.0;
		if((s[i]->t_position.y + s[i]->boundingSphere()) < min.x)
			s[i]->t_velocity.y *= -1.0;
		if((s[i]->t_position.z - s[i]->boundingSphere()) > min.x)
			s[i]->t_velocity.z *= -1.0;
	}
}



