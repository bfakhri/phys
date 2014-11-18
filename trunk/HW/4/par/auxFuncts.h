#ifndef AUXFUNCTS_H
#define AUXFUNCTS_H

#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

// Global Constants
extern const double START_A;
extern const double END_B;
extern const double EPSILON;
//extern const double EPSILON;
extern const double SLOPE;
extern const int MGR_BUFF_SIZE;
extern const int WKR_BUFF_SIZE;
extern const int DEBUG_FREQ;

// For curState of circalQueues
extern const int STATUS_EMPTY;
extern const int STATUS_MID; 
extern const int STATUS_FULL; 

// For global circalQueue 
extern const int FUN_DEQUEUE; 
extern const int FUN_SINGLE_Q;
extern const int FUN_DOUBLE_Q; 

// Global Stuff
extern bool manager_allWorking; 
extern double manager_curMaxVal; 
extern double manager_circalQueue[];
extern int manager_front; 
extern int manager_back; 
extern int manager_buffState; 
extern bool * manager_dArray;

// Function we want to find the maximum of
double mathFun(double x);

// Local Circular Queue
bool worker_qWork(double c, double d, double * circalQueue, int * front, int * back, int * curState);

bool worker_deqWork(double * c, double * d, double * circalQueue, int * front, int * back, int * curState);

// Global Circular Queue 
bool manager_safeWorkBuffer(int function, double * c, double * d, double c2, double d2);

// Gives front value but does not pop it off the queue
bool worker_peek(double * c, double * d, double * circalQueue, int * front, int * back, int * curState);

// Returns true only if max changed
bool worker_setMax(double * currentMax, double fc, double fd);

// Returns true only if max changed
bool manager_setMax(double fc, double fd);

// Returns true only if it is possible to get a higher value in this interval
bool promisingInterval(double currentMax, double c, double d);

// Attempts to rid itself of a piece of the interval handed to it
bool cutInterval(double currentMax, double * c, double * d);

// Returns space left in circalQueue 
int currentCapacity(int circalQueueSize, int front, int back, int curState);

// Returns true if all processors are done 
bool readyToLeave(bool * doneArr, int numThreads);

// Returns the amount of the remaining interval represented in the circalQueue 
// as a percentage
// FOR DEBUGGING
double intervalLeft(double originalSize, double * circalQueue, int circalQueueSize, int front, int back, int curState);

// Returns the average size of the subintervals in the circalQueue
// FOR DEBUGGING ONLY
double averageSubintervalSize(double * circalQueue, int circalQueueSize, int front, int back, int curState);

// Prints the intervals in the circalQueue
// FOR DEBUGGING ONLY
void printBuff(double * circalQueue, int circalQueueSize, int front, int back, int count);

// FOR DEBUGGING
void spinWait();

void printDiagOutput(int * d, int worker_front, int worker_back, int worker_buffState, int worker_threadNum, double * worker_circalQueue);

#endif
