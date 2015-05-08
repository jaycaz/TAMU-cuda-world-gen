// Jordan Cazamias
// CUDA World Gen 2015

#pragma once

#include <math.h>
#include <Windows.h>

#include "stats.h"

/* These define:s are for the GIF-saver... */
/* a code_int must be able to hold 2**BITS values of type int, and also -1 */
typedef int             code_int;

#ifdef SIGNED_COMPARE_SLOW
typedef unsigned long int count_int;
typedef unsigned short int count_short;
#else /*SIGNED_COMPARE_SLOW*/
typedef long int          count_int;
#endif /*SIGNED_COMPARE_SLOW*/

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
int             *WorldMapArray;
int             XRange;
int             YRange;
int             Histogram[256];
int             FilledPixels;
int             Red[49];
int             Green[49];
int             Blue[49];
float           YRangeDiv2, YRangeDivPI;
float           *SinIterPhi;



/* 4-connective floodfill algorithm which I use for constructing
*  the ice-caps.*/
void FloodFill4(int x, int y, int OldColor);
void GenerateWorldMap();