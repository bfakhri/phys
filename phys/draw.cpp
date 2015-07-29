#include "draw.h"

void drawShape(cart position, double volume) // NOT FINISHED SIGNATURE
{
	// Fill in body
}


void drawAllShapes(vector<Shape> v)
{
	for(int i=0; i<v.size(); i++){
		v[i].draw();
	}
}
