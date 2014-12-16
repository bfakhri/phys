#include "debug.h"

using namespace std;

// For debugging
void printTime(uint64_t seconds){
	uint64_t days =  seconds/(60*60*24);
	seconds = seconds%(60*60*24); 
	uint64_t hours = seconds/(60*60); 
	seconds = seconds%(60*60); 
	uint64_t minutes = seconds/60; 
	seconds = seconds%60; 
	cout << "Days:    " << days << "\tHrs:    " << hours << "\tMin:   " << minutes << "\tSec:   " << seconds << "\t"; 
}

void printF(Mass m){
	cout << "Forces:\tX = " << m.getCumalForces().x << "\tY = " << m.getCumalForces().y << "\tZ = " << m.getCumalForces().z << endl; 
} 
void printVel(Mass m){
	cout << "Velocities:\tX = " << m.getVelocity().x << "\tY = " << m.getVelocity().y << "\tZ = " << m.getVelocity().z; 
} 
void printPos(Mass m){
	cout << "Pos:\tX = " << m.getPos().x << "   \tY = " << m.getPos().y << "   \tZ = " << m.getPos().z; 
} 
void printDist(Mass m1, Mass m2){
	cartesian diffPos;
	diffPos.x =  m1.getPos().x - m2.getPos().x;
	diffPos.y =  m1.getPos().y - m2.getPos().y;
	diffPos.z =  m1.getPos().z - m2.getPos().z;
	double distance = sqrt(diffPos.x*diffPos.x + diffPos.y*diffPos.y + diffPos.z*diffPos.z); 
	cout << "\t   R = " << distance << endl; 
}
