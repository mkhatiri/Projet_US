#include <cuda_runtime_api.h>
#include <iostream>
#include <list>
#include "tbb/concurrent_queue.h"
#include "math.h"
#include "timestamp.hpp"
#define  nThreadsPerBlocks 512 


struct stream_container {
	int id;
	int *d_a;
	int *d_b;
	int *d_c;
	int beginning;
	int size;
	int iteration;
	int device;
	//const long long int nBlock;
	cudaStream_t stream;

};

struct Task {  
	int id;
	int beginning;
	int size;
};

tbb::concurrent_bounded_queue<stream_container*> streams;
std::list<Task> tasks;
Task medium_tasks;
void callkernel(stream_container *stream);
void update_stream_liste(stream_container *stream);
void split_input_to_tasks(long long int size);

void add_new_idle_stream(stream_container *stream) {
	streams.push(stream);
}

void update_stream_list(tbb::concurrent_bounded_queue<stream_container*>& streams, stream_container *stream){
	add_new_idle_stream(stream);
}

__device__ void device_function(int index, int *d_a, int *d_b, int *d_c){
	int res = 1 ;
	for(int i = 0 ; i < d_a[index]*100; i++)
		res += d_b[index];

	d_c[index] = res;

}
// TODO tommorow
__global__ void calcule_V(int *d_a, int *d_b, int *d_c, int beginning, int subsize) {
	long long int index = ((long long int)blockIdx.x) * blockDim.x + threadIdx.x + beginning;

	int end = beginning + subsize  ;

	if (index < end && index >= beginning) { 

		//		for(int i=0; i < 10 ; i++)
		device_function(index, d_a, d_b, d_c);
	}
}

__global__ void add_V(int *d_a, int *d_b, int *d_c, int beginning, int subsize) {
	long long int index = ((long long int)blockIdx.x) * blockDim.x + threadIdx.x + beginning;

	int end = beginning + subsize  ;

	if (index < end && index >= beginning) { 
		d_c[index] = d_a[index] + d_b[index]  ;
	}
}

void cudaPrintError(std::string m) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cerr<<m<<" : "<<cudaGetErrorString(err)<<std::endl;
	}
}

void CUDART_CB call_back(cudaStream_t Stream, cudaError_t err, void* data){

	stream_container *stream = (stream_container *) data;
	//std::cout << "  ---------- stream " << stream->id << " on GPU " << stream->device<< " terminer task wish started at " << stream->beginning <<  std::endl;

	cudaPrintError("in call_back_befor update stream list");
	update_stream_list(streams, stream);
	cudaPrintError("in call_back after update stream list");

}

void callkernel(stream_container *stream){

	const long long int nBlocks = (stream->size / nThreadsPerBlocks) + ( (stream->size % nThreadsPerBlocks) == 0 ? 0 : 1);

	add_V <<<nBlocks, nThreadsPerBlocks,0,stream->stream>>>(stream->d_a,stream->d_b,stream->d_c,stream->beginning, stream->size);

	cudaPrintError("after kernel 76 \n");
	cudaStream_t stream_ = stream->stream;	
	cudaStreamAddCallback(stream_,call_back,stream,0);
	cudaPrintError("after callback 95 \n");

}

void call_strong_kernel(stream_container *stream){

	const long long int nBlocks = (stream->size / nThreadsPerBlocks) + ( (stream->size % nThreadsPerBlocks) == 0 ? 0 : 1);

	calcule_V <<<nBlocks, nThreadsPerBlocks,0,stream->stream>>>(stream->d_a,stream->d_b,stream->d_c,stream->beginning, stream->size);

	cudaPrintError("after kernel hard 110 \n");
	cudaStream_t stream_ = stream->stream;	
	cudaStreamAddCallback(stream_,call_back,stream,0);
	cudaPrintError("after callback hard 113 \n");

}


Task get_task(int device){
	Task t;
	if(device == 0){
		t = tasks.front();
		tasks.pop_front();
		//	std::cout << " get task " << t.id << "(" << t.beginning << "," << t.size << ") for device 0 " << std::endl; 
		//return t;
	}else{
		t = tasks.back();
		tasks.pop_back();
		//	std::cout << " get task " << t.id << "(" << t.beginning << "," << t.size << ") for device 1 " << std::endl;

		//return t;
	}

	return t;
}

void split_input_to_tasks(long long int size, int subsize ){

	int id = 0;
	for(int i=0 ; i < size ; i+=subsize){
		Task t;        
		t.id = id++;	
		if( i + subsize <= size){
			t.beginning = i;
			t.size = subsize;	 
			tasks.push_back(t);
		}else{
			int subsize = size - i ;
			t.beginning = i;
			t.size = subsize;	 
			tasks.push_back(t);
		}        
	}

	//	std::cout << " nombre de task " << id << std::endl;

}

void init_stream(int* d_a0, int* d_b0, int* d_c0, int* d_a1, int* d_b1, int* d_c1, int stream_number){


	for(int i=0 ; i < stream_number ; i++ ) {

		cudaSetDevice(0);
		stream_container * stream;
		stream = (stream_container *) malloc(sizeof(stream_container));

		stream->id = i;
		stream->d_a = d_a0;
		stream->d_b = d_b0;
		stream->d_c = d_c0;
		stream->device = 0;
		cudaStream_t new_stream0;
		cudaStreamCreate( &new_stream0 );

		stream->stream = new_stream0;
		add_new_idle_stream(stream);

		cudaSetDevice(1);
		stream = (stream_container *) malloc(sizeof(stream_container));

		stream->id = i;
		stream->d_a = d_a1;
		stream->d_b = d_b1;
		stream->d_c = d_c1;
		stream->device = 1;
		cudaStream_t new_stream1;
		cudaStreamCreate( &new_stream1 );

		stream->stream = new_stream1;
		add_new_idle_stream(stream);

		//		std::cout << " creat stream : " << i << std::endl;
	}




}

void send_data_to_GPU(int*& a,int*& b, int*& c, int*& d_a0, int*& d_b0, int*& d_c0, int*& d_a1, int*& d_b1, int*& d_c1, long long int size){

	a = (int *) malloc(size * sizeof(int)); 
	b = (int *) malloc(size * sizeof(int)); 
	c = (int *) malloc(size * sizeof(int));


	for(int i=0; i< size; i++){
		a[i] = 2;
		b[i] = 2;
		c[i] = 0;
	}

	cudaSetDevice(0);

	cudaMalloc((void**)&d_a0, size*sizeof(int)) ;
	cudaMalloc((void**)&d_b0, size*sizeof(int)) ;
	cudaMalloc((void**)&d_c0, size*sizeof(int)) ;


	cudaMemcpy(d_a0, a, size*sizeof(int), cudaMemcpyHostToDevice);
	//	std::cout << " send first vector to GPU : " << std::endl;

	cudaMemcpy(d_b0, b, size*sizeof(int), cudaMemcpyHostToDevice);
	//	std::cout << " send seconde vector to GPU : " << std::endl;

//	cudaMemcpy(d_c0, c, size*sizeof(int), cudaMemcpyHostToDevice);
	//	std::cout << " send result vector to GPU : " << std::endl;

	cudaSetDevice(1);

	cudaMalloc((void**)&d_a1, size*sizeof(int)) ;
	cudaMalloc((void**)&d_b1, size*sizeof(int)) ;
	cudaMalloc((void**)&d_c1, size*sizeof(int)) ;


	cudaMemcpy(d_a1, a, size*sizeof(int), cudaMemcpyHostToDevice);
	//	std::cout << " send first vector to GPU : " << std::endl;

	cudaMemcpy(d_b1, b, size*sizeof(int), cudaMemcpyHostToDevice);
	//	std::cout << " send seconde vector to GPU : " << std::endl;

//	cudaMemcpy(d_c1, c, size*sizeof(int), cudaMemcpyHostToDevice);
	//	std::cout << " send result vector to GPU : " << std::endl;

}

void get_data_from_GPU(int* c, int* d_c0, int* d_c1,int middle, int size){


	cudaSetDevice(0);
	cudaMemcpy(c, d_c0, middle*sizeof(int), cudaMemcpyDeviceToHost);

	cudaSetDevice(1);
	cudaMemcpy((c + middle), (d_c1+middle) , (size - middle)*sizeof(int), cudaMemcpyDeviceToHost);



	std::cout << "get data from GPU " << std::endl;

//	for(int i = 0 ; i < size ; i++ )
		std::cout << "c[0] = " << c[0] << " c["<< size-1 <<"] = " << c[size-1] ;  

	std::cout<<std::endl;
}	

void put_work_on_stream(stream_container* current_stream, Task current_task, int num_kernel){

	// update info on current_stream
	cudaSetDevice(current_stream->device);
	//std::cout << "put task " << current_task.id << " on GPU " << current_stream->device << " stream "<< current_stream->id << std::endl;

	current_stream->beginning = current_task.beginning;
	current_stream->size = current_task.size;

	if(num_kernel == 0)
		callkernel(current_stream);
	else
		call_strong_kernel(current_stream);

}

void run(long long int size, int subsize, int stream_number, int test_nb, int display, int kernel){

	int *a, *b, *c;
	int *d_a0, *d_b0, *d_c0;
	int *d_a1, *d_b1, *d_c1;
	util::timestamp total(0,0);
	util::timestamp start(0,0);
	char timestr[20];
	
	send_data_to_GPU(a, b, c, d_a0, d_b0, d_c0, d_a1, d_b1, d_c1 ,size);

	std::cout << "size subsize stream_nb time" << std::endl;

//	float elapsedTime;
//	cudaEvent_t start, stop;
//	cudaEventCreate( &start );
//	cudaEventCreate( &stop );

	init_stream(d_a0, d_b0, d_c0, d_a1, d_b1, d_c1, stream_number);
	start = util::timestamp();
//	cudaEventRecord( start, 0 );
	std::cout << size << " " << subsize << " " << stream_number << " ";
	//-->	
	for (int rep = 0; rep < test_nb; rep++ ){
		split_input_to_tasks(size,subsize);	
		stream_container *current_stream;
		Task current_task;

		while(!tasks.empty()){
			streams.pop(current_stream);
			current_task = get_task(current_stream->device); 
			put_work_on_stream(current_stream, current_task, kernel);	
		}

		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaSetDevice(1);
		cudaDeviceSynchronize();
		//-->

	}
//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//	cudaEventElapsedTime( &elapsedTime, start, stop );
	
	util::timestamp stop;
	total = stop - start ;
	stop.to_c_str(timestr, 20);
	
	std::cout << total/test_nb << " \n";

	stream_container *current_stream;
	streams.pop(current_stream);
	while(current_stream->device != 1){
		streams.pop(current_stream);
	}


	if(display == 1 ){
		std::cout<< "last stream on device 1 started at " << current_stream->beginning << std::endl; 
		int middle = current_stream->beginning;
		get_data_from_GPU(c, d_c0, d_c1, middle, size);
	}

	cudaSetDevice(0);
	cudaFree(d_a0);
	cudaFree(d_b0);
	cudaFree(d_c0);

	cudaSetDevice(1);
	cudaFree(d_a1);
	cudaFree(d_b1);
	cudaFree(d_c1);

	free(a);
	free(b);
	free(c);

}


// kernel 0 = addition
// kernel 1 = hard calcul
void Simulation(int puiss_de_2, int kernel){

	long long int size = std::pow(2,puiss_de_2);
	int *a, *b, *c;
	int *d_a0, *d_b0, *d_c0;
	int *d_a1, *d_b1, *d_c1;
	int stream_number = 1;
	int id = 0 ;
	send_data_to_GPU(a, b, c, d_a0, d_b0, d_c0, d_a1, d_b1, d_c1, size);

	std::cout << "size subsize stream_nb time" << std::endl;
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	int p_ss = 10 ;


	for(int subsize = std::pow(2, p_ss); p_ss < 24 ;  subsize = std::pow(2, ++p_ss)){
		subsize = subsize * 32 ;
		for(int i=1; i<=64; i=i*2){

			if(size/subsize >= i*2  ){

				cudaEventRecord( start, 0 );
				stream_number = i ; 

				init_stream(d_a0, d_b0, d_c0, d_a1, d_b1, d_c1, stream_number);
				std::cout << id++ << " " << size << " " << subsize << " " << stream_number << " ";
				//-->	
				int rep;	
				for (rep = 0; rep < 15; rep++ ){

					split_input_to_tasks(size,subsize);	
					stream_container *current_stream;
					Task current_task;

					while(!tasks.empty()){
						streams.pop(current_stream);
						current_task = get_task(current_stream->device); 
						put_work_on_stream(current_stream, current_task,kernel);	
					}


					cudaSetDevice(0);
					cudaDeviceSynchronize ();
					cudaSetDevice(1);
					cudaDeviceSynchronize ();

				}
				cudaEventRecord( stop, 0 );
				cudaEventSynchronize( stop );
				cudaEventElapsedTime( &elapsedTime, start, stop );
				std::cout << (elapsedTime/rep)/1000  << " ";

			}}}


	cudaSetDevice(0);
	cudaFree(d_a0);
	cudaFree(d_b0);
	cudaFree(d_c0);

	cudaSetDevice(1);
	cudaFree(d_a1);
	cudaFree(d_b1);
	cudaFree(d_c1);

	free(a);
	free(b);
	free(c);
}

int main(int argc, char* argv[]){

	if(argv[1][00] == 'S'){
		std::cout << "Simulation" << std::endl;
		int puiss = atoi(argv[2]);
		int kernel = atoi(argv[3]);
		Simulation(puiss ,kernel);	
	}else{
		long long int size = atol(argv[1]);
		long long int subsize = atol(argv[2]);
		int stream_number = atoi(argv[3]);
		int test_number = atoi(argv[4]);
		int kernel = atoi(argv[5]);
		if(argv[6] && atoi(argv[6]) == 1){
			run(size, subsize, stream_number, test_number,1,kernel);
		}else{
			run(size, subsize, stream_number, test_number,0,kernel);
		}
	}




	return 0;
}

