// Jordan Cazamias
// CUDA World Gen 2015

#pragma once

#define PI        3.141593

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


/* These define:s are for the GIF-saver... */
/* a code_int must be able to hold 2**BITS values of type int, and also -1 */
typedef int             code_int;

#ifdef SIGNED_COMPARE_SLOW
typedef unsigned long int count_int;
typedef unsigned short int count_short;
#else /*SIGNED_COMPARE_SLOW*/
typedef long int          count_int;
#endif /*SIGNED_COMPARE_SLOW*/

