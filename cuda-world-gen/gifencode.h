// Jordan Cazamias
// CUDA World Gen 2015

#pragma once

#include <stdlib.h>
#include <stdio.h>

#include "worldgen.h"

void BumpPixel(void);
int GIFNextPixel(void);
void GIFEncode(FILE* fp, int GWidth, int GHeight, int GInterlace, int Background, int BitsPerPixel, int Red[], int Green[], int Blue[]);
void Putword(int w, FILE* fp);
void compress(int init_bits, FILE* outfile);
void output(code_int code);
void cl_block(void);
void cl_hash(count_int hsize);
void writeerr(void);
void char_init(void);
void char_out(int c);
void flush_char(void);

