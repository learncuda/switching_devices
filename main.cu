#include <stdio.h>

__global__ void cube(float * d_out, float * d_in) {
	int index = threadIdx.x;
	float f = d_in[index];
	d_out[index] = f * f * f;
}

__global__ void square(float * d_out, float * d_in) {
	int index = threadIdx.x;
	float f = d_in[index];
	d_out[index] = f * f;
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 25;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// generate the input array on the host
	float h1_in[ARRAY_SIZE], h2_in[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; i++) {
		h1_in[i] = h2_in[i] = float(i);
	}
	float h1_out[ARRAY_SIZE], h2_out[ARRAY_SIZE];

	// declare GPU memory pointers
	float * d1_in, *d2_in;
	float * d1_out, *d2_out;

	cudaSetDevice(0);
	// allocate GPU memory
	cudaMalloc((void**) &d1_in, ARRAY_BYTES);
	cudaMalloc((void**) &d1_out, ARRAY_BYTES);

	cudaSetDevice(1);
	cudaMalloc((void**) &d2_in, ARRAY_BYTES);
	cudaMalloc((void**) &d2_out, ARRAY_BYTES);

	cudaSetDevice(0);
	// transfer the array to the GPU
	cudaMemcpy(d1_in, h1_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaSetDevice(1);
	cudaMemcpy(d2_in, h2_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

	cudaSetDevice(0);
	// launch the kernel
	cube<<<1, ARRAY_SIZE>>>(d1_out, d1_in);

	cudaSetDevice(1);
	square<<<1, ARRAY_SIZE>>>(d2_out, d2_in);

	cudaSetDevice(0);
	// copy back the result array to the CPU
	cudaMemcpy(h1_out, d1_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	cudaSetDevice(1);
	cudaMemcpy(h2_out, d2_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h1_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}
	printf("\n---------------------------------------------------\n\n");
	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", h2_out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaSetDevice(0);
	cudaFree(d1_in);
	cudaFree(d1_out);

	cudaSetDevice(1);
	cudaFree(d2_in);
	cudaFree(d2_out);

	return 0;
}
