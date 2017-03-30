#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MCW MPI_COMM_WORLD
#define no_of_blocks 2
#define alpha 0.2

void print_error(int e) {
	if (e == 0)
		return;
	int e_class, len;
	char str[20];
	MPI_Error_class(e, &e_class);
	MPI_Error_string(e_class, str, &len);
	str[len] = '\0';
	printf("Error : %d - %s\n", e, str);
	exit(1);
}

__global__ void computeRegression(double *x, double *y, double* w0, double *w1, int n) {

	//n: No. of datapoints for each MPI Process
	int blockNo = blockIdx.x; //The Block No. for the Thread
	int offset = (n / gridDim.x); // No. of datapoints per Block
	double x_sum = 0, x2_sum = 0, x_y_sum = 0, y_sum = 0;
	
	int i = blockNo*offset; //Each Block has access to range of blockNo*(offset) -> blockNo*offset+ offset-1
	//printf("blockNo = %d, gridDim = %d, n = %d, i = %d\n", blockNo, gridDim.x, n, i);
	for (; i < blockNo*offset + offset; i++) {
		x_sum += x[i];
		y_sum += y[i];
		x2_sum += x[i] * x[i];
		x_y_sum += x[i] * y[i];
		//printf("%d:%d)x = %.1lf, y = %.1lf, x2 = %.1lf, xy = %.1lf\n", blockNo, i, x_sum, y_sum, x2_sum, x_y_sum);
		//printf("%d) x[%d]= %.1lf, y[%d]=%.1lf\n", blockNo, i, x[i], i, y[i]);
	}
	w0[blockNo] = (x2_sum*y_sum - x_sum*x_y_sum) / (((double)offset)*x2_sum - x_sum*x_sum);
	w1[blockNo] = (-x_sum*y_sum + ((double)offset)*x_y_sum) / (((double)offset)*x2_sum - x_sum*x_sum);
	//printf("%d)x = %.1lf, y = %.1lf, x2 = %.1lf, xy = %.1lf\n", blockNo,x_sum, y_sum, x2_sum, x_y_sum);
	//printf("%d)w0 = %.1lf, w1 = %.1lf\n", blockNo,w0[blockNo], w1[blockNo]);
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int rank = 0, size = 0;
	MPI_Comm_rank(MCW, &rank);
	MPI_Comm_size(MCW, &size);
	int m = 0;
	int e = 0;
	MPI_Errhandler_set(MCW, MPI_ERRORS_RETURN);
	double *x = NULL, *y = NULL;
	if (rank == 0) {	//read data
		printf("Enter the no. of data points : ");
		fflush(stdout);
		scanf("%d", &m);	//no. of data points
		x = (double*)malloc(m * sizeof(double));
		y = (double*)malloc(m * sizeof(double));
		printf("Enter the data : ");
		fflush(stdout);
		for (int i = 0; i < m; i++) {
			scanf("%lf", &x[i]);
			scanf("%lf", &y[i]);
		}
	}
	e = MPI_Bcast(&m, 1, MPI_INT, 0, MCW);
	print_error(e);
	int n = m / size; //no. of elements that each process gets
	

	double *mapped_x = (double*)malloc(n * sizeof(double));
	double *mapped_y = (double*)malloc(n * sizeof(double));

	//scatter x and y
	MPI_Barrier(MCW);
	MPI_Scatter(x, n, MPI_DOUBLE, mapped_x, n, MPI_DOUBLE, 0, MCW);
	MPI_Scatter(y, n, MPI_DOUBLE, mapped_y, n, MPI_DOUBLE, 0, MCW);

	
	//send x and y to device
	double *dev_x = NULL, *dev_y = NULL;
	cudaMalloc((void**)&dev_x, n * sizeof(double));
	cudaMalloc((void**)&dev_y, n * sizeof(double));
	cudaMemcpy(dev_x, mapped_x, n * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, mapped_y, n * sizeof(double), cudaMemcpyHostToDevice);

	//set w on device
	double w0[no_of_blocks] = { 0 }, w1[no_of_blocks] = { 0 };
	double *dev_w0 = NULL, *dev_w1 = NULL;
	cudaMalloc((void**)&dev_w0, no_of_blocks * sizeof(double));
	cudaMalloc((void**)&dev_w1, no_of_blocks * sizeof(double));
	double w0_old = 0, w1_old = 0;
	do {
		//compute w
		computeRegression<<<no_of_blocks, 1>>> (dev_x, dev_y, dev_w0, dev_w1, n);

		//get w from device
		cudaMemcpy(w0, dev_w0, no_of_blocks * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(w1, dev_w1, no_of_blocks * sizeof(double), cudaMemcpyDeviceToHost);


		//reduce all approximations of w
		double w0_avg = 0, w1_avg = 0;
		for (int i = 0; i < no_of_blocks; i++) {
			w0_avg += w0[i];
			w1_avg += w1[i];
			printf("Host : w0[%d] = %lf, w1[%d] = %lf\n", i, w0[i], i, w1[i]);
		}
		w0_avg /= (double)no_of_blocks;
		w1_avg /= (double)no_of_blocks;

		//compute w from individual approximations of each process
		double w0_avg_avg = 0, w1_avg_avg = 0;
		MPI_Reduce(&w0_avg, &w0_avg_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MCW);
		MPI_Reduce(&w1_avg, &w1_avg_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MCW);

		if (rank == 0) {
			//Iterate?
			w0_avg_avg /= (double)size;
			w1_avg_avg /= (double)size;
			printf("w0 = %.4lf, w1 = %.4lf\nDo you want to optimize further? 1/0 : ", w0_avg_avg, w1_avg_avg);
			fflush(stdout);
			int choice;
			scanf("%d", &choice);
			if (!choice) {
				w0_old = w0_old*alpha + (1 - alpha) * w0_avg_avg;
				w1_old = w1_old*alpha + (1 - alpha) * w1_avg_avg;
				printf("FINAL : w0 = %.4lf, w1 = %.4lf\n", w0_old, w1_old);

				printf("\nExiting!\n\n");
				fflush(stdout);
				MPI_Abort(MCW, 0);
				//MPI_Finalize();
				//exit(0);
			}

			//read new batch of data
			printf("Enter the new data : ");
			for (int i = 0; i < m; i++) {
				scanf("%lf", &x[i]);
				scanf("%lf", &y[i]);
			}
		}
		MPI_Barrier(MCW);
		//scatter the new data
		MPI_Scatter(x, n, MPI_DOUBLE, mapped_x, n, MPI_DOUBLE, 0, MCW);
		MPI_Scatter(y, n, MPI_DOUBLE, mapped_y, n, MPI_DOUBLE, 0, MCW);
		cudaMemcpy(dev_x, mapped_x, n * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y, mapped_y, n * sizeof(double), cudaMemcpyHostToDevice);
		
		//set new approximation of w on device
		for (int i = 0; i < no_of_blocks; i++) {
			w0[i] = w0_avg_avg;
			w1[i] = w1_avg_avg;
		}
		w0_old = w0_old*alpha + (1-alpha) * w0_avg_avg;
		w1_old = w1_old*alpha + (1-alpha) * w1_avg_avg;
		printf("current : w0 = %.4lf, w1 = %.4lf\n", w0_old, w1_old);
	} while (1);
	//MPI_Finalize();
}