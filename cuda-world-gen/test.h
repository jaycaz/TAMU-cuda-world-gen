// Jordan Cazamias
// CUDA World Gen 2015

#pragma once

#include <string.h>

#include "worldgen_seq.h"
#include "worldgen_pll.cuh"

#define DEFAULT_NUM_FAULTS 5000
#define DEFAULT_NUM_TRIALS 25

// Print average times over a certain number of trials
void run_seq_trials(int numTrials, int numFaults);
void run_pll_trials(int numTrials, int numFaults);

// Run a full data sweep, printing average times over several trial sets
void collect_seq_data(int numSets, int *trials, int *faults);
void collect_pll_data(int numSets, int *trials, int *faults);