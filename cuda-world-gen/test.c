// Jordan Cazamias
// CUDA World Gen 2015

#include "test.h"

char* timestamp()
{
	time_t curtime;
	time(&curtime);

	char *strtime = (char*)malloc(128);
	sprintf_s(strtime, 128, "%s", ctime(&curtime));

	// Replace spaces with underscores, colons with hyphens
	for (int i = 0; i < strlen(strtime); i++)
	{
		char c = strtime[i];
		//printf("searching timestamp, %c\n", c);
		if (c == ' ')
		{
			strtime[i] = '_';
		}
		else if (c == ':')
		{
			strtime[i] = '-';
		}
		else if (c == '\n')
		{
			strtime[i] = '_';
		}
	}

	return strtime;
}

void collect_seq_data(int numSets, int *trials, int *faults)
{
	// Open CSV file to store data
	FILE *fp;
	char filename[128];
	sprintf_s(filename, 128, "seq_%s.csv", timestamp());
	fp = fopen(filename, "w");

	// Print headers
	printf("Sequential\n");
	printf("%s", seq_headers());
	fprintf(fp, "%s", seq_headers());

	// Run trials
	for (int i = 0; i < numSets; i++)
	{
		run_seq_trials(trials[i], faults[i]);
		printf(seq_data(faults[i]));
		fprintf(fp, seq_data(faults[i]));
	}

	// Close file
	fclose(fp);
	return;
}

void collect_pll_data(int numSets, int *trials, int *faults)
{
	// Open CSV file to store data
	FILE *fp;
	char filename[128];
	sprintf_s(filename, 128, "pll_%s.csv", timestamp());
	fp = fopen(filename, "w");

	// Print headers
	printf("Parallel\n");
	printf("%s", pll_headers());
	fprintf(fp, "%s", pll_headers());

	// Run trials
	for (int i = 0; i < numSets; i++)
	{
		run_pll_trials(trials[i], faults[i]);
		printf(pll_data(faults[i]));
		fprintf(fp, pll_data(faults[i]));
	}

	// Close file
	fclose(fp);

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
