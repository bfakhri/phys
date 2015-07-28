// Contains references to all of the global variables and constants etc

// All globally necessary libraries
#include <vector.h>


// Defines the cartesian structure
typedef struct{
	double x; 
	double y;
	double z; 
}cart;


// Vector containing all "normal" shapes
// Declared in shapes.h file
extern vector <Shapes>allShapes; 
