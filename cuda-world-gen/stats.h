// Jordan Cazamias
// CUDA World Gen 2015
#pragma once

//#include <sys/time.h>
#include <Windows.h>

// Sequential algorithm timing variables
long seq_total_usec; // Total sequential algorithm time
long seq_rng_usec; // Total time for random number generation component
long seq_comp_usec; // Total time for staple computations, esp. elevation change
long seq_color_usec; // Process of coloring the map
long seq_gif_usec; // Saving data out to GIF file

// Given start and end time, in ticks, give output, in microseconds
long get_elapsed_usec(LARGE_INTEGER start, LARGE_INTEGER end);

void print_seq_stats();
