/*
Meta-Iterative Map-Reduce to perform massively parallel Regression with MPI and CUDA.
-------------------
About: CUDA-aware MPI program to perform [Meta] Iterative Map-Reduce for performing [Linear] Regression massively parallely.
Authors: Shikhar Srivastava and Jawahar Reddy

Details:
	~ CUDA-aware MPI: Accelerate MPI by leveraging GPU compute through CUDA. https://devblogs.nvidia.com/introduction-cuda-aware-mpi/
	~ Iterative MapReduce : The Map-reduce paradigm was adapted for iterative operations, for example in Machine Learning model training. https://deeplearning4j.org/iterativereduce
	~ [Meta] Iterative MapReduce: We (the authors) proposed a model that performs two "levels" of iterative map-reduce operations. It's gist is that each map-operation in the first level of map-reduce is a composite of another level of map-reduce operation. < Efficiency bounds are better this way >
	~ [Linear] Regression: We train a Linear regression model as proof-of-concept to showcase the Meta Iterative Map-reduce paradigm.

------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MCW MPI_COMM_WORLD
#define no_of_blocks 2
#define alpha 0.2

// Print readable error for MPI-process  
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

// CUDA Kernel code (GPU) to learnt parameters of linear regression model (CUDA Kernel model-training code)

__global__ void computeRegression(double *x, double *y, double* w0, double *w1, int n) {

	// n: No. of datapoints for each MPI Process
		int blockNo = blockIdx.x; //The Block No. for the Thread
		int offset = (n / gridDim.x); // No. of datapoints per Block
		double x_sum = 0, x2_sum = 0, x_y_sum = 0, y_sum = 0;
	
	// Each Block has access to [blockNo*(offset), blockNo*(offset)+ (offset-1)] range of indices in the data-array
		int i = blockNo*offset; 
	
	// Computing x, x^2, y, y^2 for the above range of data indices 
		for (; i < blockNo*offset + offset; i++) {

			x_sum += x[i];
			y_sum += y[i];
			x2_sum += x[i] * x[i];
			x_y_sum += x[i] * y[i];
		}
	// For current iteration, computing weights w1,wo for the linear equation y = w1.x + w0.	

		w0[blockNo] = (x2_sum*y_sum - x_sum*x_y_sum) / (((double)offset)*x2_sum - x_sum*x_sum);
		w1[blockNo] = (-x_sum*y_sum + ((double)offset)*x_y_sum) / (((double)offset)*x2_sum - x_sum*x_sum);
	
	//Debugging prints for checking thread outputs 

		//printf("%d:%d)x = %.1lf, y = %.1lf, x2 = %.1lf, xy = %.1lf\n", blockNo, i, x_sum, y_sum, x2_sum, x_y_sum);
		//printf("%d) x[%d]= %.1lf, y[%d]=%.1lf\n", blockNo, i, x[i], i, y[i]);
		//printf("blockNo = %d, gridDim = %d, n = %d, i = %d\n", blockNo, gridDim.x, n, i);
		//printf("%d)x = %.1lf, y = %.1lf, x2 = %.1lf, xy = %.1lf\n", blockNo,x_sum, y_sum, x2_sum, x_y_sum);
		//printf("%d)w0 = %.1lf, w1 = %.1lf\n", blockNo,w0[blockNo], w1[blockNo]);
}

// MPI Main program (CPU)

int main(int argc, char* argv[]) {

	// Initialize MPI Program 
		MPI_Init(&argc, &argv);
	// Get rank and size of the process inside the MPI process communication group
		int rank = 0, size = 0;
		MPI_Comm_rank(MCW, &rank);
		MPI_Comm_size(MCW, &size);
		int m = 0;
		int e = 0;
	// Set Error-handler for process
		MPI_Errhandler_set(MCW, MPI_ERRORS_RETURN);
	// Create pointers to training data for our regression model, simple format: (input x, output y)
		double *x = NULL, *y = NULL;

	// For base program (with rank = 0), we allocate training data memory and read the data.
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
	// Broadcast the success code (1) to all process in the MPI Communication group
		e = MPI_Bcast(&m, 1, MPI_INT, 0, MCW);
		print_error(e); // Print readable error if any
		int n = m / size; //no. of data elements that each process gets
	

	double *mapped_x = (double*)malloc(n * sizeof(double));
	double *mapped_y = (double*)malloc(n * sizeof(double));

	//Primary Map Operation: Scatter x and y training data to MPI Processes in groups of 'n'
		MPI_Barrier(MCW); // All MPI processes arrive at Barrier synchronously
		MPI_Scatter(x, n, MPI_DOUBLE, mapped_x, n, MPI_DOUBLE, 0, MCW);
		MPI_Scatter(y, n, MPI_DOUBLE, mapped_y, n, MPI_DOUBLE, 0, MCW);

	
	// Now for each MPI Process: Allocate memory for received batch of training data (x,y) to respective GPU block
		double *dev_x = NULL, *dev_y = NULL;
		cudaMalloc((void**)&dev_x, n * sizeof(double));
		cudaMalloc((void**)&dev_y, n * sizeof(double));
		cudaMemcpy(dev_x, mapped_x, n * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y, mapped_y, n * sizeof(double), cudaMemcpyHostToDevice);

	// Set weights 'w' of linear regression model (y = w1.x + w0) on GPU device
		double w0[no_of_blocks] = { 0 }, w1[no_of_blocks] = { 0 }; //initialization
		double *dev_w0 = NULL, *dev_w1 = NULL; // copy of 'w' for computation and manipulation
		cudaMalloc((void**)&dev_w0, no_of_blocks * sizeof(double));
		cudaMalloc((void**)&dev_w1, no_of_blocks * sizeof(double));
		double w0_old = 0, w1_old = 0;

	// Perform Iterative (Meta) Map-Reduce until error-converges	
		do { 
			// Secondary Map Operation: Assign data to threads in GPU, compute weights for all threads 'j' in GPU
 
				// Call Kernel Code : Compute Regression weights 'w' in MPI Process 'i' and CUDA GPU 'j' out of 'n'
				computeRegression<<<no_of_blocks, 1>>> (dev_x, dev_y, dev_w0, dev_w1, n);

				// Receive weights 'w' after new iteration from GPU device
				cudaMemcpy(w0, dev_w0, no_of_blocks * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(w1, dev_w1, no_of_blocks * sizeof(double), cudaMemcpyDeviceToHost);


			// Secondary Reduce operation: Reduce all approximations of 'w' from GPU onto MPI Process 'i'
				
				double w0_avg = 0, w1_avg = 0;
				// Sum up all weights
				for (int i = 0; i < no_of_blocks; i++) {
					w0_avg += w0[i];
					w1_avg += w1[i];
					printf("Host : w0[%d] = %lf, w1[%d] = %lf\n", i, w0[i], i, w1[i]);
				}
				// Compute Averages 
				w0_avg /= (double)no_of_blocks;
				w1_avg /= (double)no_of_blocks;

				// Compute w from individual approximations of each process
				double w0_avg_avg = 0, w1_avg_avg = 0;
				
			// Primary Reduce operation: Reduce all approximations of 'w' from all 'i' MPI Processes onto root '0'th MPI Process
  
				// Reduce operation (sum) on weights w0,w1 back to 0th MPI Process
				MPI_Reduce(&w0_avg, &w0_avg_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MCW);
				MPI_Reduce(&w1_avg, &w1_avg_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MCW);
			
			// For Root Process (MPI Process 0)
			// Checks if regression error has converged or more optimization of weights (model training) is required.
				if (rank == 0) {
					
					// Simple command line checks with the user
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
						
					}

					// Read new batch of training data
					printf("Enter the new data : ");
					for (int i = 0; i < m; i++) {
						scanf("%lf", &x[i]);
						scanf("%lf", &y[i]);
					}
				}

			
				MPI_Barrier(MCW); // All MPI Processes execute and arrive here synchronously. 
				
			// Primary Map Operation: Scatter the new data (x,y) to the 'i' MPI Processes

				MPI_Scatter(x, n, MPI_DOUBLE, mapped_x, n, MPI_DOUBLE, 0, MCW);
				MPI_Scatter(y, n, MPI_DOUBLE, mapped_y, n, MPI_DOUBLE, 0, MCW);
				cudaMemcpy(dev_x, mapped_x, n * sizeof(double), cudaMemcpyHostToDevice);
				cudaMemcpy(dev_y, mapped_y, n * sizeof(double), cudaMemcpyHostToDevice);
		
			// For each MPI Process 'i': Set new avg approximation of 'w' on GPU devices

				for (int i = 0; i < no_of_blocks; i++) {
					w0[i] = w0_avg_avg;
					w1[i] = w1_avg_avg;
				}
				w0_old = w0_old*alpha + (1-alpha) * w0_avg_avg;
				w1_old = w1_old*alpha + (1-alpha) * w1_avg_avg;
				printf("current : w0 = %.4lf, w1 = %.4lf\n", w0_old, w1_old);
		} while (1);

	MPI_Finalize(); // MPI Process execution ends. Child processes are deallocated.
}
