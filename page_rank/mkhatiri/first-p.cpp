#include <stdio.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include "math.h"
#include "streamUtils.hpp"
#include "tbb/concurrent_queue.h"
using namespace std;


int main (int argc, char* argv[] ) {

	int nb_block = atoi(argv[1]);
	int nVtx = atoi(argv[2]);
//	int nVtx;
	int xadj[] = {0,2,4,7,11,14};
	float val [] = {10,-2,3,9,7,8,7,3,8,7,5,8,9,13};
	int adj [] = {0,4,0,1,1,2,3,0,2,3,4,1,3,4};
	float prin [] = {0.4,4.0,1.1,2.3,0.2};
	float prout [] = {0.1,0.2,0.3,0.4,0.5};
	int subsize = 2;
	list<Task<int,int,float> > *tasks = new  list<Task<int,int,float> >;

	int X;
	X = (int) nVtx/(nb_block*2) ;
	X = X / 32 ;
	X = (X+1) * 32;

	float  nbr = (float) nVtx/X;
	cout << "---------- " << X << " ------" << nbr << "-------------" << endl;





/*	int nb = split_input_to_tasks<int, int, float>(xadj, nVtx, subsize, prin, prout, *tasks);
	cout << "---------- " << nb << " -------------------" << endl;
	tbb::concurrent_bounded_queue<stream_container<int,int,float>* >* streams;
	streams = new tbb::concurrent_bounded_queue<stream_container<int,int,float>* >;

	tbb::concurrent_bounded_queue<stream_container<int,int,float>* >* streams1;
	
	float alpha = 0.5;
	creat_stream<int, int, float>(5, &alpha, &alpha, val, xadj, adj, prin, prout, streams, 5 ); 
	
	int a = 0;	
* 	while(a < 1){
	stream_container<int, int, float> *current_stream;
	Task <int,int,float> t = get_task<int,int,float>(tasks, a++);
	streams->pop(current_stream);
	put_work_on_stream<int,int,float>(current_stream,t);
	cout << "id : ------------------ "<< current_stream->id << " ---------------------------------" << endl;
	cout << "	m : "<< current_stream->m << endl;
	cout << "	n : "<< current_stream->n << endl;
	cout << "	nnz : "<< current_stream->nnz << endl;
	cout << "	alpha : "<< *current_stream->alpha << endl;
	cout << "	beta : "<< *current_stream->beta << endl;
	cout << "	Val : "<< *current_stream->d_val << endl;
	cout << "	xdaj : "<< *current_stream->d_xadj << endl;
	cout << "	adj : "<< *current_stream->d_adj << endl;
	cout << "	prin: "<< *current_stream->d_prin << endl;
	cout << "	prout : "<< *current_stream->d_prout << endl;
	}
i*
//	get_task(*tasks);
//	get_task(*tasks);
	stream_container<int, int, float> *current_stream;
        streams->pop(current_stream);

	
	streams1 = current_stream->streams;
	a = 0; 
	cout << "------------------ " <<endl;
	while(++a < 15){
        	stream_container<int, int, float> *current_stream1;
        	streams1->pop(current_stream1);
        	cout << "stream : "<< current_stream1->id << endl;
        }

	a = 0;
	while(a < tasks->size()){
		Task <int,int,float> t = get_task<int,int,float>(tasks, a++);
	//	tasks->pop_front();
		cout << a << " - > id " << t.id << ", nVtx : " << t.nVtx << ", nnz : " << t.nnz << ", RowPtr : " << t.RowPtr << endl;
	}
*/
}
