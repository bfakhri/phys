 #include "all_includes.h"

double Shape::getX(){
	return xCord; 
}

double Shape::getY(){
	return yCord; 
}

double Shape::getRotAngle(){
	return rotAngle; 
}

double Shape::getRevAngle(){
	return revAngle; 
}

double Shape::getRelX(){
	return (xCord*cos(revAngle*PI/180));
}

double Shape::getRelY(){
	return (xCord*sin(revAngle*PI/180));
}

void Shape::setX(double c){
	xCord = c;
}

void Shape::setY(double c){
	yCord = c;
}

void Shape::setRotAngle(double angle){
	rotAngle = angle;
}

void Shape::setRevAngle(double angle){
	revAngle = angle;
}

Shape::Shape(){
	xCord = 0;
	yCord = 0;
	rotAngle = 0; 
	revAngle = 0; 
}

Shape::Shape(double x, double y, double angle){
	xCord = x; 
	yCord = y;
	rotAngle = angle; 
}

Rectangle::Rectangle(double x, double y, double rWidth, double rHeight, double rotAng, double revAng)
{
	xCord = x; 
	yCord = y; 
	height = rHeight;
	width = rWidth; 
	rotAngle = rotAng; 
	revAngle = revAng; 
}

int Rectangle::draw()
{
	glPushMatrix(); 
	glRotatef(revAngle, 0, 0 , 1); 
	glTranslatef(xCord, yCord, 0); 
	glRotatef(-revAngle, 0, 0 , 1);
	glRotatef(rotAngle, 0, 0 , 1); 
	int retInt = drawRectangle(width, height, ext_filled);
	glPopMatrix(); 
	return retInt; 
}

double Rectangle::getBoundingRadius()
{
	return sqrt((width/2)*(width/2)+(height/2)*(height/2)); 
}

Circle::Circle()
{
	xCord = 0; 
	yCord = 0; 
	revAngle = 0;
	radius = 0;
}

Circle::Circle(double x, double y, double revAng, double r)
{
	xCord = x; 
	yCord = y; 
	revAngle = revAng;
	radius = r;
}

void Circle::draw()
{
	glPushMatrix(); 
	glRotatef(revAngle, 0, 0 , 1); 
	glTranslatef(xCord, yCord, 0); 
	drawCircle(radius, DEF_SAMPLES, ext_filled);
	glPopMatrix(); 
}

double Circle::getBoundingRadius()
{
	return radius; 
}

Triangle::Triangle(double x, double y, double baseWidth, double rotAng, double revAng)
{
	xCord = x; 
	yCord = y;
	width = baseWidth; 
	height = width*sin(PI/3);
	peakToCenter = width*sin(PI/6)/sin(2*PI/3);
	rotAngle = rotAng; 
	revAngle = revAng; 
}

double Triangle::getWidth()
{
	return width; 
}

double Triangle::getHeight()
{
	return height; 
}

void Triangle::draw()
{
	glPushMatrix(); 
	glRotatef(revAngle, 0, 0 , 1); 
	glTranslatef(xCord, yCord, 0); 
	glRotatef(-revAngle, 0, 0 , 1); 
	glRotatef(rotAngle, 0, 0 , 1);
	drawTriangle(-width/2, (peakToCenter-height), width/2, (peakToCenter-height), 0, peakToCenter, ext_filled); 
	glPopMatrix(); 
}

double Triangle::getBoundingRadius()
{
	return sqrt((width/2)*(width/2)+(height-peakToCenter)*(height-peakToCenter));
}


Missile::Missile()
{
	xCord = 0; 
	yCord = 0; 
	revAngle = 0;
	radius = 0;
	xVelocity = 0;
	yVelocity = 0; 
	visible = true; 
}

Missile::Missile(double x, double y, double revAng, double r)
{
	xCord = x; 
	yCord = y; 
	revAngle = revAng;
	radius = r;
	xVelocity = 0;
	yVelocity = 0; 
	visible = true; 
}

void Missile::setXVelocity(double vel)
{
	xVelocity = vel; 
}

void Missile::setYVelocity(double vel)
{
	yVelocity = vel;
}

double Missile::getXVelocity()
{
	return xVelocity;
}

double Missile::getYVelocity()
{
	return yVelocity;
}