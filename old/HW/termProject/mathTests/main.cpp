#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mather.h"

using namespace std;

int main()
{
	double G = scientificNotation(6.67384, -11);
	//printf("G in hex is: %1.20f\n", G); 
	//float G2 = (float)G; 
	//uint32_t * G3ptr = (uint32_t*)(&G2); 
	//printf("G in hex is: %#X\n", *G3ptr); 
	//int L = 0x11;
	//printf("G in hex is: %#X\n", ((uint32_t)L)); 
	
	double test = 0x3dd2584c222c163e; 
	cout << "Test: " << test << endl;
	cout << "G: " << G << endl;
	cout << "SN: " << scientificNotation(6.67384, -11) << endl;
	
	cout << "G: " << G << endl;
	
	uint64_t dest; 

	memcpy((void*)&dest, (void*)&G, 8); 	

	printf("G in hex is: %#lX\n", dest);

	double G2 = 0X3DD2584C222C163E;   
0X3DD2584C222C163E
	cout << "G2: " << G2 << endl; 
	return 0; 
}
