#include <iostream>
#include <unistd.h>
#include <string>
#include <list>
#include <stdio.h>
#include <string.h>
#include "tbb/concurrent_queue.h"
#include <stdlib.h>
#include "streamUtils.hpp"
#include <cstdlib>
#include <iterator>

using namespace std;




template <typename VertexType, typename EdgeType, typename Scalar>
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, stream_container <VertexType, EdgeType, Scalar>* stream) {
	streams->push(stream);
}


template <typename VertexType, typename EdgeType, typename Scalar>
int split_input_to_tasks(EdgeType* xadj, VertexType nVtx, VertexType subsize, Scalar* d_prin, Scalar* d_prout, list<Task<VertexType, EdgeType, Scalar> >& tasks) { 

	int id = 0;
	EdgeType lnnz = 0;
	for(VertexType i = 0; i < nVtx; i += subsize){
		Task<VertexType, EdgeType, Scalar> t;        
		t.id = id++;	
		if(i <= nVtx - subsize){
			if(i == 0){
				t.RowPtr = i  ;
				t.nnz = xadj[subsize];
				lnnz = xadj[subsize];
				t.xadj = xadj;	
				t.prout = d_prout ;
				t.prin = d_prin;

			}else{
				t.RowPtr = i ;
				t.nnz = xadj[i + subsize] - lnnz ;
				lnnz = xadj[i + subsize];
				t.xadj = xadj + i;
				t.prout = d_prout + i;
				t.prin = d_prin + i;
			}

			t.nVtx = subsize ;
			//	cout << "RowPtr : " << t.RowPtr << " , nnz : " << t.nnz << " , m : "<< t.nVtx << endl; 		
			tasks.push_back(t);
		}else{
			VertexType subsize = 0; 	// - i ;
			t.RowPtr = i  ;
			t.nnz = xadj[nVtx] - lnnz;
			lnnz = xadj[nVtx];
			t.nVtx = nVtx - i ;
			t.xadj = xadj + i;
			t.prout = d_prout + i;
			t.prin = d_prin + i;
			//	cout << "RowPtr : " << t.RowPtr << " , nnz : " << t.nnz << " , m : "<< t.nVtx << endl; 		
			//t.beginning = i;
			//t.size = subsize;	 
			tasks.push_back(t);
		}        
	}
	return id;
	//	cout << "lnnx total" << lnnz << endl;


}

template <typename VertexType, typename EdgeType, typename Scalar>
Task<VertexType, EdgeType, Scalar> get_task(list<Task<VertexType, EdgeType, Scalar> >* tasks,int index) {

	//Task<VertexType, EdgeType, Scalar> t = tasks.front();
	//tasks.pop_front();
	//template <typename VertexType, typename EdgeType, typename Scalar>
	typename list<Task<VertexType, EdgeType, Scalar> >::iterator it = tasks->begin();
	advance(it, index);

	return *it;
}


template <typename VertexType, typename EdgeType, typename Scalar>
void creat_stream(VertexType nVtx, Scalar* alpha, Scalar* beta, Scalar* d_val, EdgeType* d_xadj, VertexType *d_adj, Scalar* d_prin, Scalar* d_prout, tbb::concurrent_bounded_queue<stream_container<VertexType, EdgeType, Scalar>*>* streams, int stream_number ) {

	for(int i=0 ; i < stream_number ; i++ ) {

		stream_container<VertexType, EdgeType, Scalar> * stream;
		stream = (stream_container<VertexType, EdgeType, Scalar> * ) malloc(1*sizeof(stream_container<VertexType, EdgeType, Scalar>));

		stream->id = i;
		stream->n = nVtx;
		stream->alpha = alpha;
		stream->beta = beta;
		stream->d_val = d_val;
		stream->d_xadj = d_xadj;
		stream->d_adj = d_adj;
		stream->d_prin = d_prin;
		stream->d_prout = d_prout;
		stream->streams = streams;
		cudaStream_t new_stream;
		cudaStreamCreate( &new_stream );
		stream->stream = new_stream;
		//add_new_idle_stream(stream);
		//streams->push(stream);
		add_new_idle_stream<VertexType, EdgeType, Scalar>(streams, stream);
		//		std::cout << " creat stream : " << i << std::endl;
	}
}

template <typename VertexType, typename EdgeType, typename Scalar>
void put_work_on_stream(stream_container<VertexType, EdgeType, Scalar>* current_stream, Task<VertexType, EdgeType, Scalar> current_task){

	current_stream->m = current_task.nVtx;
	current_stream->nnz = current_task.nnz;
	current_stream->d_xadj = current_task.xadj;
	current_stream->d_prin = current_task.prin;
	current_stream->d_prout = current_task.prout;
}



template 
void add_new_idle_stream(tbb::concurrent_bounded_queue<stream_container<int,int,float>*>* streams, stream_container <int,int,float>* stream);

template 
int split_input_to_tasks<int,int,float>(int* xadj, int nVtx, int subsize, float* d_prin, float* d_prout, list<Task<int, int, float> >& tasks);

template 
Task<int,int,float> get_task(list<Task<int,int,float> >* tasks, int index);

template
void creat_stream(int nVtx, float* alpha, float* beta, float* d_val, int* d_xadj, int *d_adj, float* d_prin, float* d_prout, tbb::concurrent_bounded_queue<stream_container<int, int, float>*>* streams, int stream_number );

template 
void put_work_on_stream(stream_container<int, int, float>* current_stream, Task<int, int, float> current_task);

