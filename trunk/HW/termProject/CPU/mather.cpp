#include "mather.h"

double scientificNotation(double num,  int exp)
{ 
	if(exp > 0)
	{
		for(int i=0; i<exp; i++){
			num *= 10; 
		}
	}else{
		for(int i=0; i>exp; i--){
			num /= 10; 
		}
	}
}
