#include "mather.h"

void scientificNotation(double * outNum, double num,  int exp)
{
	*outNum = num; 
	for(int i=0; i<exp; i++){
		(*outNum) *= 10; 
	}
}
