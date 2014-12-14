#ifndef SHAPEOBJECTS_H
#define SHAPEOBJECTS_H

#define DEF_SAMPLES 100
#include "all_includes.h"

// Parent shape class
class Shape
{
protected:
	double xCord;
	double yCord;
	double rotAngle;		// Rotation angle of object
	double revAngle;		// Revolution around an axis (useful when in orbit)
public:
	double getX();		// Returns X coordinate
	double getY();		// Returns Y coordinate 
	double getRotAngle();	
	double getRevAngle();
	double getRelX();		// Gets relative X value when orbiting
	double getRelY();		// Gets relative Y value when orbiting

	void setX(double c);
	void setY(double c);
	void setRotAngle(double angle);
	void setRevAngle(double angle);
	
	// Constructors
	Shape();
	Shape(double x, double y, double angle);
};

// Rectangle
class Rectangle: public Shape
{
private:
	double width; 
	double height; 
public:
	Rectangle(double x, double y, double width, double height, double rotAng, double revAng);
	int draw();
	double getBoundingRadius();	// Returns the radius of the bounding circle
								// that envelopes the rectangle
};

// Circle
class Circle: public Shape
{
protected:
	double radius; 
public:
	Circle(); 
	Circle(double x, double y, double revAngle, double r);
	void draw(); 
	double getBoundingRadius();	// Returns the radius of the bounding circle
								// that envelopes the circle (same as radius)
};

// Triangle - equilateral 
class Triangle: public Shape
{
private:
	double width; 
	double height; 
	double peakToCenter;			// Distance from the peak of the triangle to center
public:
	Triangle(double x, double y, double baseWidth, double rotAng, double revAng);
	void draw(); 
	double getWidth();
	double getHeight(); 
	double getBoundingRadius();	// Returns the radius of the bounding circle
								// that envelopes the triangle
};

// Circle that has velocity components
class Missile: public Circle
{
private: 
	double xVelocity;			// Horiz component of velocity
	double yVelocity;			// Vert component of velocity
	bool visible; 
public:
	Missile(); 
	Missile(double x, double y, double revAng, double r); 

	void setXVelocity(double vel); 
	void setYVelocity(double vel); 
	double getXVelocity();
	double getYVelocity(); 
};


// Rocket

// Cloud


#endif
