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
#define GLOBAL_BUFF_SIZE 10000
#define LOCAL_BUFF_SIZE 10000
#define DEBUG_FREQ 1000

// For status of circalQueues
#define STATUS_EMPTY 0
#define STATUS_MID 1
#define STATUS_FULL 2

// For global circalQueue 
#define FUN_DEQUEUE 0
#define FUN_SINGLE_Q 1
#define FUN_DOUBLE_Q 2


// Global Stuff
extern bool global_allWorking; 
extern double global_curMaxVal; 
extern double global_circalQueue[];
extern int global_front; 
extern int global_back; 
extern int global_buffState; 
extern bool * global_dArray;

// Function we want to find the maximum of
double mathFun(double x);

// Local Circular Queue
bool local_qWork(double c, double d, double * circalQueue, int * head, int * tail, int * status);

bool local_deqWork(double * c, double * d, double * circalQueue, int * head, int * tail, int * status);

// Global Circular Queue 
bool global_safeWorkBuffer(int function, double * c, double * d, double c2, double d2);

// Gives front value but does not pop it off the queue
bool local_peek(double * c, double * d, double * circalQueue, int * head, int * tail, int * status);

// Returns true only if max changed
bool local_setMax(double * currentMax, double fc, double fd);

// Returns true only if max changed
bool global_setMax(double fc, double fd);

// Returns true only if it is possible to get a higher value in this interval
bool validInterval(double currentMax, double c, double d);

// Attempts to rid itself of a piece of the interval handed to it
bool shrinkInterval(double currentMax, double * c, double * d);

// Returns space left in circalQueue 
int spaceLeft(int circalQueueSize, int head, int tail, int status);

// Returns true if all processors are done 
bool allDone(bool * doneArr, int size);

// Returns the amount of the remaining interval represented in the circalQueue 
// as a percentage
// FOR DEBUGGING
double intervalLeft(double originalSize, double * circalQueue, int circalQueueSize, int head, int tail, int status);

// Returns the average size of the subintervals in the circalQueue
// FOR DEBUGGING ONLY
double averageSubintervalSize(double * circalQueue, int circalQueueSize, int head, int tail, int status);

// Prints the intervals in the circalQueue
// FOR DEBUGGING ONLY
void printBuff(double * circalQueue, int circalQueueSize, int head, int tail, int count);

// FOR DEBUGGING
void spinWait();

void printDiagOutput(int * d, int local_front, int local_back, int local_buffState, int local_threadNum, double * local_circalQueue);

#endif
