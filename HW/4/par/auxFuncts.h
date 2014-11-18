#ifndef AUXFUNCTS_H
#define AUXFUNCTS_H

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <iostream>

#define START_A 1 
#define END_B 100
#define EPSILON 0.000001 // 10^-6
//#define EPSILON 0.0	// FOR DEBUGGING
#define SLOPE 12
#define MGR_BUFF_SIZE 10000
#define WKR_BUFF_SIZE 10000
#define DEBUG_FREQ 1000

// For curState of circalQueues
#define STATUS_EMPTY 0
#define STATUS_MID 1
#define STATUS_FULL 2

// For global circalQueue 
#define FUN_DEQUEUE 0
#define FUN_SINGLE_Q 1
#define FUN_DOUBLE_Q 2


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
bool validInterval(double currentMax, double c, double d);

// Attempts to rid itself of a piece of the interval handed to it
bool shrinkInterval(double currentMax, double * c, double * d);

// Returns space left in circalQueue 
int spaceLeft(int circalQueueSize, int front, int back, int curState);

// Returns true if all processors are done 
bool allDone(bool * doneArr, int size);

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
