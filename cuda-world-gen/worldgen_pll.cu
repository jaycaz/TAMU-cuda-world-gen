// Jordan Cazamias
// CUDA World Gen 2015

// Parallel implementation of worldgen_seq

extern "C"
{
	#include "worldgen_pll.cuh"
}

#define CUDA_MAX_BLOCKS 65535

#define CUDA_CALL(x) do {cudaError_t status = x; if(status !=cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__, status); \
	exit(status);}} while(0) 
#define CURAND_CALL(x) do {curandStatus_t status = x; if(status != CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__, status); \
	exit(status);}} while(0)

/* Function that generates the worldmap */
void GenerateWorldMapPll(unsigned seed, int numFaults);

extern "C" void genworld_pll(int numFaults)
{
	int       NumberOfFaults = 0, a, j, i, Color, MaxZ = 1, MinZ = -1;
	int       row, TwoColorMode = 0;
	int       index2;
	unsigned  Seed = 0;
	int       Threshold, Count;
	int       PercentWater, PercentIce, Cur;
	char SaveName[256];  /* 255 character filenames should be enough? */
	char SaveFile[256];  /* SaveName + .gif */
	FILE * Save;

	// Begin timing for parallel algorithm
	LARGE_INTEGER pll_start, pll_end;
	QueryPerformanceCounter(&pll_start);

	init_worldgen();
	reset_pll_times();

	if (WorldMapArray == NULL)
	{
		WorldMapArray = (int *)malloc(XRange*YRange*sizeof(int));
		if (WorldMapArray == NULL)
		{
			fprintf(stderr, "WorldMapArray could not be allocated.");
			exit(-1);
		}
	}

	if (SinIterPhi == NULL)
	{
		SinIterPhi = (float *)malloc(2 * XRange*sizeof(float));
		if (SinIterPhi == NULL)
		{
			fprintf(stderr, "SinIterPhi could not be allocated.");
			exit(-1);
		}
		for (i = 0; i<XRange; i++)
		{
			SinIterPhi[i] = SinIterPhi[i + XRange] = (float)sin(i * 2 * PI / XRange);
		}
	}

	/*
	fprintf(stderr, "Seed: ");
	scanf("%d", &Seed);
	fprintf(stderr, "Number of faults: ");
	scanf("%d", &NumberOfFaults);
	fprintf(stderr, "Percent water: ");
	scanf("%d", &PercentWater);
	fprintf(stderr, "Percent ice: ");
	scanf("%d", &PercentIce);

	fprintf(stderr, "Save as (.GIF will be appended): ");
	scanf("%8s", SaveName);
	*/

	Seed = time(NULL);
	NumberOfFaults = numFaults;
	PercentWater = 60;
	PercentIce = 10;
	strcpy(SaveName, "default_pll");

	srand(Seed);

	for (j = 0, row = 0; j<XRange; j++)
	{
		WorldMapArray[row] = 0;
		for (i = 1; i<YRange; i++) WorldMapArray[i + row] = INT_MIN;
		row += YRange;
	}

	/* Define some "constants" which we use frequently */
	YRangeDiv2 = (float) YRange / 2;
	YRangeDivPI = (float) YRange / PI;

	/* Generate the map! */
	// Call world generation kernel
	GenerateWorldMapPll(Seed, NumberOfFaults);

	/* Copy data (I have only calculated faults for 1/2 the image.
	* I can do this due to symmetry... :) */
	index2 = (XRange / 2)*YRange;
	for (j = 0, row = 0; j<XRange / 2; j++)
	{
		for (i = 1; i<YRange; i++)                    /* fix */
		{
			WorldMapArray[row + index2 + YRange - i] = WorldMapArray[row + i];
		}
		row += YRange;
	}

	/* Reconstruct the real WorldMap from the WorldMapArray and FaultArray */
	for (j = 0, row = 0; j<XRange; j++)
	{
		/* We have to start somewhere, and the top row was initialized to 0,
		* but it might have changed during the iterations... */
		Color = WorldMapArray[row];
		for (i = 1; i<YRange; i++)
		{
			/* We "fill" all positions with values != INT_MIN with Color */
			Cur = WorldMapArray[row + i];
			if (Cur != INT_MIN)
			{
				Color += Cur;
			}
			WorldMapArray[row + i] = Color;
		}
		row += YRange;
	}

	// Time coloring
	LARGE_INTEGER pll_color_start, pll_color_end;
	QueryPerformanceCounter(&pll_color_start);

	/* Compute MAX and MIN values in WorldMapArray */
	for (j = 0; j<XRange*YRange; j++)
	{
		Color = WorldMapArray[j];
		if (Color > MaxZ) MaxZ = Color;
		if (Color < MinZ) MinZ = Color;
	}

	/* Compute color-histogram of WorldMapArray.
	* This histogram is a very crude aproximation, since all pixels are
	* considered of the same size... I will try to change this in a
	* later version of this program. */
	for (j = 0, row = 0; j<XRange; j++)
	{
		for (i = 0; i<YRange; i++)
		{
			Color = WorldMapArray[row + i];
			Color = (int)(((float)(Color - MinZ + 1) / (float)(MaxZ - MinZ + 1)) * 30) + 1;
			Histogram[Color]++;
		}
		row += YRange;
	}

	/* Threshold now holds how many pixels PercentWater means */
	Threshold = PercentWater*XRange*YRange / 100;

	/* "Integrate" the histogram to decide where to put sea-level */
	for (j = 0, Count = 0; j<256; j++)
	{
		Count += Histogram[j];
		if (Count > Threshold) break;
	}

	/* Threshold now holds where sea-level is */
	Threshold = j*(MaxZ - MinZ + 1) / 30 + MinZ;

	if (TwoColorMode)
	{
		for (j = 0, row = 0; j<XRange; j++)
		{
			for (i = 0; i<YRange; i++)
			{
				Color = WorldMapArray[row + i];
				if (Color < Threshold)
					WorldMapArray[row + i] = 3;
				else
					WorldMapArray[row + i] = 20;
			}
			row += YRange;
		}
	}
	else
	{
		/* Scale WorldMapArray to colorrange in a way that gives you
		* a certain Ocean/Land ratio */
		for (j = 0, row = 0; j<XRange; j++)
		{
			for (i = 0; i<YRange; i++)
			{
				Color = WorldMapArray[row + i];

				if (Color < Threshold)
					Color = (int)(((float)(Color - MinZ) / (float)(Threshold - MinZ)) * 15) + 1;
				else
					Color = (int)(((float)(Color - Threshold) / (float)(MaxZ - Threshold)) * 15) + 16;

				/* Just in case... I DON't want the GIF-saver to flip out! :) */
				if (Color < 1) Color = 1;
				if (Color > 255) Color = 31;
				WorldMapArray[row + i] = Color;
			}
			row += YRange;
		}

		/* "Recycle" Threshold variable, and, eh, the variable still has something
		* like the same meaning... :) */
		Threshold = PercentIce*XRange*YRange / 100;

		if ((Threshold <= 0) || (Threshold > XRange*YRange)) goto Finished;

		FilledPixels = 0;
		/* i==y, j==x */
		for (i = 0; i<YRange; i++)
		{
			for (j = 0, row = 0; j<XRange; j++)
			{
				Color = WorldMapArray[row + i];
				//if (Color < 32) FloodFill4(j, i, Color);
				/* FilledPixels is a global variable which FloodFill4 modifies...
				* I know it's ugly, but as it is now, this is a hack! :)
				*/
				if (FilledPixels > Threshold) goto NorthPoleFinished;
				row += YRange;
			}
		}

	NorthPoleFinished:
		FilledPixels = 0;
		/* i==y, j==x */
		for (i = (YRange - 1); i>0; i--)            /* fix */
		{
			for (j = 0, row = 0; j<XRange; j++)
			{
				Color = WorldMapArray[row + i];
				//if (Color < 32) FloodFill4(j, i, Color);
				/* FilledPixels is a global variable which FloodFill4 modifies...
				* I know it's ugly, but as it is now, this is a hack! :)
				*/
				if (FilledPixels > Threshold) goto Finished;
				row += YRange;
			}
		}
	Finished:;
	}

	// Finish timing coloring
	QueryPerformanceCounter(&pll_color_end);
	pll_color_usec += get_elapsed_usec(pll_color_start, pll_color_end);

	// Start timing save to gif
	LARGE_INTEGER pll_gif_start, pll_gif_end;
	QueryPerformanceCounter(&pll_gif_start);

	/* append .gif to SaveFile */
	sprintf(SaveFile, "%s.gif", SaveName);
	/* open binary SaveFile */
	Save = fopen(SaveFile, "wb");
	/* Write GIF to savefile */

	GIFEncode(Save, XRange, YRange, 1, 0, 8, Red, Green, Blue);

	// Finish timing save to gif
	QueryPerformanceCounter(&pll_gif_end);
	pll_gif_usec += get_elapsed_usec(pll_gif_start, pll_gif_end);

	//fprintf(stderr, "Map created, saved as %s.\n", SaveFile);

	free(WorldMapArray);
	free(SinIterPhi);
	WorldMapArray = NULL;
	SinIterPhi = NULL;

	// Get total algorithm time
	QueryPerformanceCounter(&pll_end);
	pll_total_usec += get_elapsed_usec(pll_start, pll_end);

	return;
}

__global__ void GenCUDA(int *WorldMapArray, float *SinIterPhi, int *XRange, int *YRange, float *rands);


void GenerateWorldMapPll(unsigned seed, int numFaults)
{
	// Determine how many threads should be started
	//int numThreads = (int)XRange / 2;
	int numBlocks = numFaults;
	int threadsPerBlock = (int)XRange / 2;

	int *d_WorldMapArray;
	float *d_SinIterPhi;
	
	// Set up world map array for GPU
	size_t wmaSize = XRange * YRange * sizeof(int);
	//CUDA_CALL(cudaMalloc(&d_WorldMapArray, wmaSize));
	CUDA_CALL(cudaMalloc(&d_WorldMapArray, wmaSize));
	//printf("WMA last byte: %x\n", WorldMapArray[XRange * YRange - 1]);
	CUDA_CALL(cudaMemcpy(d_WorldMapArray, WorldMapArray, wmaSize, cudaMemcpyHostToDevice));

	// Set up SinIterPhi for GPU
	size_t sipSize = 2 * XRange * sizeof(float);
	CUDA_CALL(cudaMalloc(&d_SinIterPhi, sipSize));
	CUDA_CALL(cudaMemcpy(d_SinIterPhi, SinIterPhi, sipSize, cudaMemcpyHostToDevice));

	// Set up XRange, YRange
	int *d_XRange;
	int *d_YRange;
	CUDA_CALL(cudaMalloc(&d_XRange, sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_XRange, &XRange, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMalloc(&d_YRange, sizeof(int)));
	CUDA_CALL(cudaMemcpy(d_YRange, &YRange, sizeof(int), cudaMemcpyHostToDevice));

	// Begin RNG timing
	LARGE_INTEGER rng_start_time, rng_end_time;
	QueryPerformanceCounter(&rng_start_time);

	// Set up random numbers
	int numRands = 3 * numBlocks;
	float *d_rands;
	CUDA_CALL(cudaMalloc(&d_rands, sizeof(float) * numRands));

	// Create pseudo-random number generator
	curandGenerator_t gen;
	//CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT)); 
	// Set ordering (for increased performance)
	CURAND_CALL(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_SEEDED));
	// Set seed
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed)); 
	// Generate n floats on device
	CURAND_CALL(curandGenerateUniform(gen, d_rands, numRands));

	// End RNG timing
	QueryPerformanceCounter(&rng_end_time);
	pll_rng_usec += get_elapsed_usec(rng_start_time, rng_end_time);

	// Begin Comp timing
	LARGE_INTEGER comp_start, comp_end;
	QueryPerformanceCounter(&comp_start);

	// ***** Call kernel ******
	int curBlock;
	int runBlocks = min(numBlocks, CUDA_MAX_BLOCKS);
	for (curBlock = 0; curBlock < numBlocks; curBlock += runBlocks)
	{
		GenCUDA<<<runBlocks, threadsPerBlock>>>(d_WorldMapArray, d_SinIterPhi, d_XRange, d_YRange, d_rands);
	}
	cudaError_t status = cudaDeviceSynchronize();

	// End Comp timing
	QueryPerformanceCounter(&comp_end);
	pll_comp_usec += get_elapsed_usec(comp_start, comp_end);

	// Retrieve world map array data from GPU
	CUDA_CALL(cudaMemcpy(WorldMapArray, d_WorldMapArray, wmaSize, cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(d_WorldMapArray));
	CUDA_CALL(cudaFree(d_SinIterPhi));
}

__global__ void GenCUDA(int *WorldMapArray, float *SinIterPhi, int *XRange, int *YRange, float *rands)
{
	__shared__ float		 Alpha;
	__shared__ float		 Beta;
	__shared__ float         TanB;
	__shared__ int		     Xsi;
	__shared__ unsigned int  flag1;
	int			  *wma_ptr;
	int			  Phi;
	int			  Theta;

	// Calculate which Phi thread should take care of
	//int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	Phi = threadIdx.x;
	//printf("Phi = %d + (%d * %d)\n", threadIdx.x, blockIdx.x, blockDim.x);
	//printf("Thread id: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);

	if (Phi == 0)
	{
		// Extract random values
		//float rand[3];
		//rand[0] = rands[blockIdx.x * 3];
		//rand[1] = rands[blockIdx.x * 3 + 1];
		//rand[2] = rands[blockIdx.x * 3 + 2];

		flag1 = (int)(rands[blockIdx.x * 3] + 0.5);

		/* Create a random greatcircle...
		* Start with an equator and rotate it */
		Alpha = (rands[blockIdx.x * 3 + 1] - 0.5)*PI; /* Rotate around x-axis */
		Beta = (rands[blockIdx.x * 3 + 2] - 0.5)*PI; /* Rotate around y-axis */
		//printf("(flag1, Alpha, Beta): (%u, %f, %f)\n", flag1, Alpha, Beta);

		TanB = tan(acos(cos(Alpha)*cos(Beta)));
		Xsi = (int)((*XRange) / 2 - ((*XRange) / PI) * Beta);
		//printf("Xsi: %d\n", Xsi);
	}

	__syncthreads();
	//printf("XRange, YRange: %d, %d\n", *XRange, *YRange);

	//for (Phi = 0; Phi < XRange / 2; Phi++)
	//{
		//float YRangeDivPI = (*YRange) / PI;
		//float YRangeDiv2 = (*YRange) / 2;
		//printf("pll (siniterphi, sin) = (%f, %f)\n", SinIterPhi[Xsi - Phi + (*XRange)], sin((Xsi - Phi) * 2 * PI / (*XRange)));
		//int row = (*YRange) * Phi;
		//printf("pll_row: %d\n", row);
		Theta = (int)(((*YRange) / 2) * atan(SinIterPhi[Xsi - Phi + (*XRange)] * TanB)) + ((*YRange) / 2);
		//printf("Phi, sip, theta: %d, %f, %d\n", Phi, SinIterPhi[Xsi - Phi + (*XRange)], Theta);
		wma_ptr = WorldMapArray + ((*YRange) * Phi + Theta);

		atomicCAS(wma_ptr, INT_MIN, 0);
		if (flag1)
		{
			/* Rise northen hemisphere <=> lower southern */
			atomicSub(wma_ptr, 1);
		}
		else
		{
			/* Rise southern hemisphere */
			atomicAdd(wma_ptr, 1);
		}
	//}
}
