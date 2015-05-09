// Jordan Cazamias
// CUDA World Gen 2015

#include "worldgen.h"

extern int XRange = 320;
extern int YRange = 160;
extern int Red[49] = { 0, 0, 0, 0, 0, 0, 0, 0, 34, 68, 102, 119, 136, 153, 170, 187, 0, 34, 34, 119, 187, 255, 238, 221, 204, 187, 170, 153, 136, 119, 85, 68, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175 };
extern int Green[49] = { 0, 0, 17, 51, 85, 119, 153, 204, 221, 238, 255, 255, 255, 255, 255, 255, 68, 102, 136, 170, 221, 187, 170, 136, 136, 102, 85, 85, 68, 51, 51, 34, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175 };
extern int Blue[49] = { 0, 68, 102, 136, 170, 187, 221, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 34, 34, 34, 34, 34, 34, 34, 34, 34, 17, 0, 255, 250, 245, 240, 235, 230, 225, 220, 215, 210, 205, 200, 195, 190, 185, 180, 175 };

void init_worldgen()
{
	WorldMapArray = NULL;
	SinIterPhi = NULL;
	memset(Histogram, 0, 256);
	FilledPixels = 0;
	YRangeDiv2 = YRange / 2;
	YRangeDivPI = YRange / PI;
}

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
