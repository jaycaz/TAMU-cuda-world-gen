// Jordan Cazamias
// CUDA World Gen 2015
#pragma once

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "worldgen.h"

extern void genworld_pll(int numFaults);
