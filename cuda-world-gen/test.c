// Jordan Cazamias
// CUDA World Gen 2015

#include "test.h"

void run_seq_trials(int numTrials, int numFaults)
{
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
