// Jordan Cazamias
// CUDA World Gen 2015

//#include <sys/time.h>
#include <Windows.h>

#include "stats.h"
#include "worldgen_seq.h"
//#include "worldgen_pll.cuh"

extern void genworld_pll(int argc, char **argv);

int main(int argc, char **argv)
{
	LARGE_INTEGER seq_start, seq_end;
	QueryPerformanceCounter(&seq_start);

	// Run sequential generation algorithm!
	genworld_seq(argc, argv);

	// Get total algorithm time
	QueryPerformanceCounter(&seq_end);
	seq_total_usec = get_elapsed_usec(seq_start, seq_end);

	print_seq_stats();
	return 0;
}