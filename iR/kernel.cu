#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MCW MPI_COMM_WORLD

__global__ void someFunction(int *x) {

}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MCW, &rank);
	MPI_Comm_size(MCW, &size);
	if (rank == 0) {

	}

	//scatter data
	int* x;
	cudaMalloc((void**)&x, 10);
	someFunction<<<1,1>>>(x);
}