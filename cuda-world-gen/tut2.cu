// Jordan Cazamias
// CUDA World Gen 2015

#include <iostream>
#include <ctime>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

using namespace std;

__global__ void AddInts(int *a, int *b, int count)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < count)
	{
		a[id] += b[id];
	}
	//printf("id: %d\n", id);
}

/*
int main()
{
	srand(time(NULL));
	int count = 100;
	int *h_a = new int[count];
	int *h_b = new int[count];

	for (int i = 0; i < count; i++)
	{
		h_a[i] = rand() % 1000;
		h_b[i] = rand() % 1000;
	}

	cout << "Prior to addition:" << endl;
	for (int i = 0; i < 5; i++)
	{
		cout << i << ": " << h_a[i] << " " << h_b[i] << endl;
	}


	int *d_a, *d_b;
	if (cudaMalloc(&d_a, sizeof(int) * count) != cudaSuccess)
	{
		cout << "CUDA Malloc failed!";
		return 1;
	}

	if (cudaMalloc(&d_b, sizeof(int) * count) != cudaSuccess)
	{
		cout << "CUDA Malloc failed!";
		cudaFree(d_a);
		return 1;
	}

	if (cudaMemcpy(d_a, h_a, sizeof(int)*count, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "CUDA copy to device failed!";
		cudaFree(d_a);
		cudaFree(d_b);
		return 1;
	}

	if (cudaMemcpy(d_b, h_b, sizeof(int)*count, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cout << "CUDA copy to device failed!";
		cudaFree(d_a);
		cudaFree(d_b);
		return 1;
	}

	// Add integers together
	int blocks = count / 256 + 1;
	int threads = 256;
	AddInts<<<blocks, threads>>>(d_a, d_b, count);


	if (cudaMemcpy(h_a, d_a, sizeof(int)*count, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cout << "CUDA copy to host failed!";
		cudaFree(d_a);
		cudaFree(d_b);
		return 1;
	}

	if (cudaMemcpy(h_a, d_a, sizeof(int)*count, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cout << "CUDA copy to host failed!";
		cudaFree(d_a);
		cudaFree(d_b);
		return 1;
	}

	for (int i = 0; i < 5; i++)
	{
		cout << "Ans: " << h_a[i] << endl;
	}

	delete[] h_a;
	delete[] h_b;

	return 0;
}
*/