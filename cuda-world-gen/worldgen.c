// Jordan Cazamias
// CUDA World Gen 2015

#include "worldgen.h"

XRange = 320;
YRange = 160;
Red[49] = { 0, 0, 0, 0, 0, 0, 0, 0, 34, 68, 102, 119, 136, 153, 170, 187, 0, 34, 34, 119, 187, 255, 238, 221, 204, 187, 170, 153, 136, 119, 85, 68, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175 };
Green[49] = { 0, 0, 17, 51, 85, 119, 153, 204, 221, 238, 255, 255, 255, 255, 255, 255, 68, 102, 136, 170, 221, 187, 170, 136, 136, 102, 85, 85, 68, 51, 51, 34, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175 };
Blue[49] = { 0, 68, 102, 136, 170, 187, 221, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 34, 34, 34, 34, 34, 34, 34, 34, 34, 17, 0, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175 };

void FloodFill4(int x, int y, int OldColor)
{
	if (WorldMapArray[x*YRange + y] == OldColor)
	{
		if (WorldMapArray[x*YRange + y] < 16)
			WorldMapArray[x*YRange + y] = 32;
		else
			WorldMapArray[x*YRange + y] += 17;

		FilledPixels++;
		if (y - 1 > 0)      FloodFill4(x, y - 1, OldColor);
		if (y + 1 < YRange) FloodFill4(x, y + 1, OldColor);
		if (x - 1 < 0)
			FloodFill4(XRange - 1, y, OldColor);        /* fix */
		else
			FloodFill4(x - 1, y, OldColor);

		if (x + 1 >= XRange)                          /* fix */
			FloodFill4(0, y, OldColor);
		else
			FloodFill4(x + 1, y, OldColor);
	}
}

void GenerateWorldMap()
{
	float         Alpha, Beta;
	float         TanB;
	float         Result, Delta;
	int           i, row, N2;
	int           Theta, Phi, Xsi;
	unsigned int  flag1;


	/* I have to do this because of a bug in rand() in Solaris 1...
	* Here's what the man-page says:
	*
	* "The low bits of the numbers generated are not  very  random;
	*  use  the  middle  bits.  In particular the lowest bit alter-
	*  nates between 0 and 1."
	*
	* So I can't optimize this, but you might if you don't have the
	* same bug... */

	// Begin RNG timing
	LARGE_INTEGER rng_start_time, rng_end_time;
	QueryPerformanceCounter(&rng_start_time);

	flag1 = rand() & 1; /*(int)((((float) rand())/MAX_RAND) + 0.5);*/

	/* Create a random greatcircle...
	* Start with an equator and rotate it */

	Alpha = (((float)rand()) / MAX_RAND - 0.5)*PI; /* Rotate around x-axis */
	Beta = (((float)rand()) / MAX_RAND - 0.5)*PI; /* Rotate around y-axis */

	// End RNG timing
	QueryPerformanceCounter(&rng_end_time);
	seq_rng_usec += get_elapsed_usec(rng_start_time, rng_end_time);


	// Begin comp timing
	LARGE_INTEGER comp_start_time, comp_end_time;
	QueryPerformanceCounter(&comp_start_time);

	TanB = tan(acos(cos(Alpha)*cos(Beta)));
	row = 0;
	Xsi = (int)(XRange / 2 - (XRange / PI)*Beta);

	for (Phi = 0; Phi<XRange / 2; Phi++)
	{
		Theta = (int)(YRangeDivPI*atan(*(SinIterPhi + Xsi - Phi + XRange)*TanB)) + YRangeDiv2;
		//printf("theta: %d, ", Theta);

		if (flag1)
		{
			/* Rise northen hemisphere <=> lower southern */
			if (WorldMapArray[row + Theta] != INT_MIN)
				WorldMapArray[row + Theta]--;
			else
				WorldMapArray[row + Theta] = -1;
		}
		else
		{
			/* Rise southern hemisphere */
			if (WorldMapArray[row + Theta] != INT_MIN)
				WorldMapArray[row + Theta]++;
			else
				WorldMapArray[row + Theta] = 1;
		}

		//printf("row: %d\n", row);
		row += YRange;
	}

	// End comp time
	QueryPerformanceCounter(&comp_end_time);
	seq_comp_usec += get_elapsed_usec(comp_start_time, comp_end_time);
}
