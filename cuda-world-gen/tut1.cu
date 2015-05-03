// Jordan Cazamias
// CUDA World Gen

// tut1.cu: tutorial 1 cude file

#include <iostream>

#include "cuda_runtime.h"

using namespace std;

__global__ void AddIntsCUDA(int *a, int *b)
{
	a[0] += b[0];
}

/*
int main()
{
	int a = 5, b = 9;
	int *d_a, *d_b;

	cudaMalloc(&d_a, sizeof(int));
	cudaMalloc(&d_b, sizeof(int));

	cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

	// Call kernel on GPU
	AddIntsCUDA<<<1, 1>>>(d_a, d_b);

	cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "The answer is " << a << endl;

	cudaFree(d_a);
	cudaFree(d_b);

	return 0;
}
*/