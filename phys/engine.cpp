#include "engine.h"



void physicsThread(std::vector<Shape*> v)
{
	while(1)
	{
		advanceSim(SIM_T, v);
		enforceBoundaries(v, physBoundaryMin, physBoundaryMin);
	}
}
