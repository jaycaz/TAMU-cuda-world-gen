// Jordan Cazamias
// CUDA World Gen 2015

#include "test.h"

void collect_seq_data(int numSets, int *trials, int *faults)
{
	printf("Sequential\n");
	printf("%s", seq_headers());
	for (int i = 0; i < numSets; i++)
	{
		run_seq_trials(trials[i], faults[i]);
		printf(seq_data(faults[i]));
	}
	return;
}

void collect_pll_data(int numSets, int *trials, int *faults)
{
	printf("Parallel\n");
	printf("%s", pll_headers());
	for (int i = 0; i < numSets; i++)
	{
		run_pll_trials(trials[i], faults[i]);
		printf(pll_data(faults[i]));
	}
	return;
}

void run_seq_trials(int numTrials, int numFaults)
{
	reset_seq_times();

	for (int i = 0; i < numTrials; i++)
	{
		genworld_seq(numFaults);
	}

	seq_total_usec /= numTrials;
	seq_rng_usec /= numTrials;
	seq_comp_usec /= numTrials;
	seq_color_usec /= numTrials;
	seq_gif_usec /= numTrials;

	return;
}

void run_pll_trials(int numTrials, int numFaults)
{
	reset_pll_times();

	for (int i = 0; i < numTrials; i++)
	{
		genworld_pll(numFaults);
	}

	pll_total_usec /= numTrials;
	pll_rng_usec /= numTrials;
	pll_comp_usec /= numTrials;
	pll_color_usec /= numTrials;
	pll_gif_usec /= numTrials;
}
