// Jordan Cazamias
// CUDA World Gen 2015

#pragma once

#include "worldgen_seq.h"
#include "worldgen_pll.cuh"

void run_seq_trials(int numTrials, int numFaults);
void run_pll_trials(int numTrials, int numFaults);