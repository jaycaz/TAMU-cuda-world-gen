// Jordan Cazamias
// CUDA World Gen 2015
#pragma once

//#include <sys/time.h>
#include <Windows.h>

// Sequential algorithm timing variables
extern long seq_total_usec; // Total sequential algorithm time
extern long seq_rng_usec; // Total time for random number generation component
extern long seq_comp_usec; // Total time for staple computations, esp. elevation change
extern long seq_color_usec; // Process of coloring the map
extern long seq_gif_usec; // Saving data out to GIF file

// Same variables, but for parallel implementation
extern long pll_total_usec;
extern long pll_rng_usec;
extern long pll_comp_usec;
extern long pll_color_usec;
extern long pll_gif_usec;

// Reset times
void reset_seq_times();
void reset_pll_times();

// Given start and end time, in ticks, give output, in microseconds
long get_elapsed_usec(LARGE_INTEGER start, LARGE_INTEGER end);

// Print verbose timing stats, for debugging
void print_seq_stats();
void print_pll_stats();

// Compact stats strings for storage in csv
char* seq_headers();
char* pll_headers();
char* seq_data(int numFaults);
char* pll_data(int numFaults);