// Jordan Cazamias
// CUDA World Gen 2015

#include <cuda_runtime.h>
#include <Windows.h>

#include "stats.h"
#include "worldgen_seq.h"
#include "worldgen_pll.cuh"
#include "test.h"

//extern void genworld_pll(int argc, char **argv);

int main(int argc, char **argv)
{
	int seq_faults[10] = { 200, 700, 2000, 7000, 20000, 70000, 200000, 700000, 2000000, 7000000 };
	int seq_trials[10] = { 50, 50, 50, 50, 30, 30, 10, 10, 5, 5 };

	int pll_faults[10] = { 200, 700, 2000, 7000, 20000, 70000, 200000, 700000, 2000000, 7000000 };
	int pll_trials[10] = { 50, 50, 50, 50, 30, 30, 10, 10, 5, 5 };

	collect_seq_data(10, seq_trials, seq_faults);
	collect_pll_data(10, pll_trials, pll_faults);

	/*
	int numFaults = 500000;
	int numTrials = 1;

	// ********* SEQUENTIAL ********** //
	// Run sequential generation algorithm
	printf("Running sequential algorithm with %d trials...", numTrials);
	//run_seq_trials(numTrials, numFaults);
	printf("done.\n");


	// ********* PARALLEL ********** //
	// Run parallel generation algorithm
	printf("Running parallel algorithm with %d trials...", numTrials);
	run_pll_trials(numTrials, numFaults);
	printf("done.\n");

	print_seq_stats();
	print_pll_stats();
	*/

	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		printf("Press Enter to continue...");
		getchar();
        return 1;
    }

	free(WorldMapArray);

	printf("Press Enter to continue...");
	//getchar();
	return 0;
}