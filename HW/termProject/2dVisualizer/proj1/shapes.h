#ifndef SHAPES_H
#define SHAPES_H
#include "all_includes.h"

#define PI 3.14159265359 
int drawRectangle(float width, float height, bool filled);
void drawCircle(float radius, unsigned int samples, bool filled); 
void drawPartialCircle(float radius, float startAngle, float endAngle, unsigned int samples); 
void drawTriangle(float x1, float y1, float x2, float y2, float x3, float y3, bool filled); 
int drawRocket(float maxHeight, float maxWidth); 
void drawCloud(float maxWidth, unsigned int samples); 

#endif
