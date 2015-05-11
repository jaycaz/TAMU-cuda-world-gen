#include <stdio.h>

#include "stats.h"

extern long seq_total_usec = 0;
extern long seq_rng_usec = 0;
extern long seq_comp_usec = 0;
extern long seq_color_usec = 0;
extern long seq_gif_usec = 0;

extern long pll_total_usec = 0;
extern long pll_rng_usec = 0;
extern long pll_comp_usec = 0;
extern long pll_color_usec = 0;
extern long pll_gif_usec = 0;

LONGLONG TICKS_PER_SEC = -1;

long get_elapsed_usec(LARGE_INTEGER start, LARGE_INTEGER end)
{
	LONGLONG elapsed = end.QuadPart - start.QuadPart;
	if (TICKS_PER_SEC == -1)
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		TICKS_PER_SEC = freq.QuadPart;
	}
	double seconds = (double)elapsed / TICKS_PER_SEC;

	//if (start.tv_usec > end.tv_usec)
	//{
	//	end.tv_usec += 1000000;
	//	end.tv_sec--;
	//}
	//elapsed.tv_usec = end.tv_usec - start.tv_usec;
	//elapsed.tv_sec = end.tv_sec - start.tv_sec;

	long elapsedusec = (long)(seconds * 1000000);

	return elapsedusec;
}

void reset_seq_times()
{
	seq_total_usec = 0;
	seq_rng_usec = 0;
	seq_comp_usec = 0;
	seq_color_usec = 0;
	seq_gif_usec = 0;
}

void reset_pll_times()
{
	pll_total_usec = 0;
	pll_rng_usec = 0;
	pll_comp_usec = 0;
	pll_color_usec = 0;
	pll_gif_usec = 0;
}

void print_seq_stats()
{
	printf("\nSequential Algorithm Statistics:\n");
	printf("\t- Total time: %ld usec\n", seq_total_usec);
	printf("\t- Elevation computation time: %ld usec\n", seq_comp_usec);
	printf("\t- RNG time: %ld usec\n", seq_rng_usec);
	printf("\t- Color time: %ld usec\n", seq_color_usec);
	printf("\t- GIF time: %ld usec\n", seq_gif_usec);
	printf("\n");
}

void print_pll_stats()
{
	printf("\nParallel Algorithm Statistics:\n");
	printf("\t- Total time: %ld usec\n", pll_total_usec);
	printf("\t- Elevation computation time: %ld usec\n", pll_comp_usec);
	printf("\t- RNG time: %ld usec\n", pll_rng_usec);
	printf("\t- Color time: %ld usec\n", pll_color_usec);
	printf("\t- GIF time: %ld usec\n", pll_gif_usec);
	printf("\n");
}

char* seq_headers()
{
	char *headers = (char*)malloc(128);
	sprintf(headers, "# Faults, Total time, Elev. Comp., RNG, Color, GIF\n");
	return headers;
}

char* pll_headers()
{
	return seq_headers();
}

char* seq_data(int numFaults)
{
	char *data = (char*)malloc(128);
	sprintf(data, "%d, %ld, %ld, %ld, %ld, %ld\n", 
			numFaults,
			seq_total_usec,
			seq_comp_usec,
			seq_rng_usec,
			seq_color_usec,
			seq_gif_usec);
	return data;
}
 
char* pll_data(int numFaults)
{
	char *data = (char*)malloc(128);
	sprintf(data, "%d, %ld, %ld, %ld, %ld, %ld\n", 
			numFaults,
			pll_total_usec,
			pll_comp_usec,
			pll_rng_usec,
			pll_color_usec,
			pll_gif_usec);
	return data;
}
