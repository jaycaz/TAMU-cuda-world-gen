// Jordan Cazamias
// CUDA World Gen 2015

#include <cuda_runtime.h>
#include <Windows.h>

#include "stats.h"
#include "worldgen_seq.h"
//#include "worldgen_pll.cuh"

extern void genworld_pll(int argc, char **argv);

int main(int argc, char **argv)
{
	// ********* SEQUENTIAL ********** //
	// Begin timing for sequential algorithm
	LARGE_INTEGER seq_start, seq_end;
	QueryPerformanceCounter(&seq_start);

	// Run sequential generation algorithm
	genworld_seq(argc, argv);
	genworld_seq(argc, argv);

	// Get total algorithm time
	QueryPerformanceCounter(&seq_end);
	seq_total_usec = get_elapsed_usec(seq_start, seq_end);

	print_seq_stats();


	// ********* PARALLEL ********** //
	// Begin timing for parallel algorithm
	LARGE_INTEGER pll_start, pll_end;
	QueryPerformanceCounter(&pll_start);

	// Run parallel generation algorithm
	//genworld_pll(argc, argv);

	// Get total algorithm time
	QueryPerformanceCounter(&pll_end);
	pll_total_usec = get_elapsed_usec(pll_start, pll_end);

	print_pll_stats();

	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		printf("Press Enter to continue...");
		getchar();
        return 1;
    }

	free(WorldMapArray);

	printf("Press Enter to continue...");
	getchar();
	return 0;
}