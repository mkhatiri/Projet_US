#include <cuda_runtime_api.h>
#include <iostream>
#include <list>
#include "tbb/concurrent_queue.h"
#include "math.h"
#define  nThreadsPerBlocks 512 

__global__ void assign(int *d_a, int *d_b, int size) {
	int index = threadIdx.x;

	if (index < size) { 
		d_b[index] = d_a[0]  ;
	}
}

void cudaPrintError(std::string m) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr<<m<<" : "<<cudaGetErrorString(err)<<std::endl;
	}
}


int main(int argc, char* argv[]){

	int GPU1 = 0;
	int GPU2 = 1;
	int size = 200;

	int *a;
	int *d_a0;

	int *b, *c;
	int *d_b1;

	cudaSetDevice(GPU1);
	a = (int *) malloc(1 * sizeof(int)); 
	a[0] = 15 ; 
	cudaMalloc((void**)&d_a0, 1*sizeof(int)) ;
	cudaMemcpy(d_a0, a, 1*sizeof(int), cudaMemcpyHostToDevice);

	cudaPrintError("after GPU1");
	cudaDeviceEnablePeerAccess(GPU1,0);

	cudaSetDevice(GPU2);	
	cudaMalloc((void**)&d_b1, size*sizeof(int)) ;
	cudaDeviceEnablePeerAccess(GPU1,0);
	cudaPrintError("after GPU2");
	
	assign<<<1, nThreadsPerBlocks>>>(d_a0,d_b1,size);
	cudaPrintError("after KERNEL");
	
	c = (int *) malloc(size * sizeof(int)); 
	cudaMemcpy(c, d_b1, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaPrintError("after getData");
	
	
	for(int i=0; i<size; i++ )
		std::cout << i << " " << c[i] << std::endl;


	cudaFree(d_a0);
	cudaFree(d_b1);
	return 0;
}
