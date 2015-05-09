// Jordan Cazamias
// CUDA World Gen 2015

#pragma once

#include    <limits.h>
#include    <stdio.h>
#include    <stdlib.h>
#include    <Windows.h>
#include    <math.h>

#include "gifencode.h"
#include "stats.h"

/* My own definitions */

#ifndef PI
#define PI        3.141593
#endif

/* This value holds the maximum value rand() can generate
*
* RAND_MAX *might* be defined in stdlib.h, if it's not
* you *might* have to change the definition of MAX_RAND...
*/
#ifdef RAND_MAX
#define MAX_RAND  RAND_MAX
#else
#define MAX_RAND  0x7FFFFFFF
#endif

// Global Variables
int             Histogram[256];
int             FilledPixels;
float           YRangeDiv2, YRangeDivPI;

extern int             *WorldMapArray;
extern float           *SinIterPhi;
extern int             XRange;
extern int             YRange;
extern int             Red[49];
extern int             Green[49];
extern int             Blue[49];

/* 4-connective floodfill algorithm which I use for constructing
*  the ice-caps.*/
void FloodFill4(int x, int y, int OldColor);
