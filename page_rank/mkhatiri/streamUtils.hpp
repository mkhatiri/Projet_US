#ifndef STREAM_UTILS_HPP
#define STREAM_UTILS_HPP

#include <stdio.h>
#include "tbb/concurrent_queue.h"
#include <cuda_runtime_api.h>
#include <list>

using namespace std;

template <typename VertexType, typename EdgeType, typename Scalar>
struct stream_container {
	int id;
	VertexType m;
	VertexType n;
	EdgeType nnz;
	Scalar* alpha;
	Scalar* beta;
	Scalar* d_val;
	EdgeType* d_xadj;
	VertexType* d_adj ;
	Scalar* d_prin;
	Scalar* d_prout;
	tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*> * streams;
	cudaStream_t stream;

};

template <typename VertexType, typename EdgeType, typename Scalar>
struct Task {  
	int id;
	VertexType nVtx;
	EdgeType nnz;
	EdgeType* xadj;
	Scalar* prin;
        Scalar* prout;
	VertexType RowPtr;
	//EdgeType* d_xadj;
};


template <typename VertexType, typename EdgeType, typename Scalar>
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, stream_container<VertexType, EdgeType, Scalar>* stream) ;


template <typename VertexType, typename EdgeType, typename Scalar>
int split_input_to_tasks(EdgeType* xadj, VertexType nVtx, VertexType subsize, Scalar* d_prin, Scalar* f_prout, list<Task<VertexType, EdgeType, Scalar> >& tasks);


template <typename VertexType, typename EdgeType, typename Scalar>
Task<VertexType, EdgeType, Scalar> get_task(list<Task<VertexType, EdgeType, Scalar> >* tasks, int index);

template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream(VertexType nVtx, Scalar* alpha, Scalar* beta, Scalar* d_val, EdgeType* d_xadj, VertexType *d_adj, Scalar* d_prin, Scalar* d_prout, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>*  streams, int stream_number ); 

template <typename VertexType, typename EdgeType, typename Scalar>
void put_work_on_stream(stream_container<VertexType, EdgeType, Scalar>* current_stream, Task<VertexType, EdgeType, Scalar> current_task);

#endif
