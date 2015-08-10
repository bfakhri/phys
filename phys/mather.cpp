#include "mather.h"

cart operator+(cart lhs, cart rhs)
{
	cart ret;
	ret.x = lhs.x + rhs.x;
	ret.y = lhs.y + rhs.y;
	ret.z = lhs.z + rhs.z;
	return ret; 
}

cart operator*(double scalar, cart rhs)
{
	cart ret;
	ret.x = rhs.x*scalar;
	ret.y = rhs.y*scalar;
	ret.z = rhs.z*scalar;
	return ret;
}

cart operator*(cart lhs, double scalar)
{
	cart ret;
	ret.x = lhs.x*scalar;
	ret.y = lhs.y*scalar;
	ret.z = lhs.z*scalar;
	return ret;
}

cart operator/(cart lhs, double scalar)
{
	double inv = 1/scalar;
	return lhs*inv; 
}
