// This file declares math related elements used throughout the project

#ifndef MATHER_H
#define MATHER_H

// This struct defines the cartesian data type (x, y, z)

typedef struct cart{
	double x;
	double y;
	double z;
}cart;

// Add two vectors
cart operator+(cart lhs, cart rhs);

// Multiply vectors by a scalar
cart operator*(double scalar, cart rhs);
cart operator*(cart lhs, double scalar);
// Divide vector by a scalar
cart operator/(cart lhs, double scalar);


#endif
