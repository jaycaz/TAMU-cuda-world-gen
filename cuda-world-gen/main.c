// Jordan Cazamias
// CUDA World Gen 2015

#include <stdio.h>
#include <cuda_runtime.h>
#include <Windows.h>

#include "stats.h"
#include "worldgen_seq.h"
#include "worldgen_pll.cuh"
#include "test.h"

//extern void genworld_pll(int argc, char **argv);

int main(int argc, char **argv)
{
	// Uncomment for full trials with data storage
	/*
	int seq_faults[10] = { 200, 700, 2000, 7000, 20000, 70000, 200000, 700000, 2000000, 7000000 };
	int seq_trials[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	int pll_faults[10] = { 200, 700, 2000, 7000, 20000, 70000, 200000, 700000, 2000000, 7000000 };
	int pll_trials[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	collect_seq_data(10, seq_trials, seq_faults);
	collect_pll_data(10, pll_trials, pll_faults);
	*/

	// Start interactive mode
	printf("Welcome to CUDA World Gen!\n");

	printf("Select world generation algorithm: [s]equential or [p]arallel: ");
	int parallel = 0;
	int c = 0;
	do
	{
		c = getchar();
	} while (c != 's' && c != 'p');
	if (c == 'p')
	{
		parallel = 1;
	}

	printf("Select number of iterations (or [d]efault = %d): ", DEFAULT_NUM_FAULTS);
	char line[128];
	int numFaults = 0;
	while (numFaults <= 0)
	{
		int charsRead = scanf("%s", line);
		if (charsRead == 1 && line[0] == 'd')
		{
			numFaults = DEFAULT_NUM_FAULTS;
			break;
		}
		sscanf(line, "%d", &numFaults);
		//numFaults = atoi(line);
	}

	printf("Select number of trials (or [d]efault = %d): ", DEFAULT_NUM_TRIALS);
	int numTrials = 0;
	while (numTrials <= 0)
	{
		int charsRead = scanf("%s", line);
		if (charsRead == 1 && line[0] == 'd')
		{
			numTrials = DEFAULT_NUM_TRIALS;
			break;
		}
		sscanf(line, "%d", &numTrials);
	}

	// Run test
	if (parallel)
	{
		printf("Running Parallel, %d faults, %d trials\n", numFaults, numTrials);
		run_pll_trials(numTrials, numFaults);
		print_pll_stats();
	}
	else
	{
		printf("Running Sequential, %d faults, %d trials\n", numFaults, numTrials);
		run_seq_trials(numTrials, numFaults);
		print_seq_stats();
	}

	cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
		printf("Press Enter to continue...");
		getchar();
        return 1;
    }

	printf("Press Enter to continue...");
	getchar();
	getchar();
	return 0;
}