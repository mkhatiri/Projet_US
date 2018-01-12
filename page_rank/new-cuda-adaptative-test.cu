#include "main-pr.hpp"

#define THROW_AWAY 0
#include "Padded2DArray.hpp"
#include <omp.h>
#include "memutils.hpp"
#include <cmath>

//#define SHOWLOADBALANCE
#include "logged_array.hpp"

//#define LOG
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "helper_cuda.h"
#include "math.h"
//#include "streamUtils.hpp"
#include "tbb/concurrent_queue.h"
#include "math.h"
#include "AdaptativeUtils.hpp"

	template <typename VertexType, typename EdgeType, typename Scalar>
int main_pr(VertexType nVtx, EdgeType* xadj_, VertexType *adj_, Scalar* val_, Scalar *prior_, Scalar* pr_,
		Scalar lambda,
		int nTry, //algo parameter
		util::timestamp& totaltime, std::string& 
	   )
{


	int subsize = 0;
	int nb_blocks = 0;
	int stream_number = 6;
/*
	{
		char* str = getenv ("NBBLOCK");
		if (str) {
			std::stringstream ss (str);
			ss>>nb_blocks;
			if (!ss)
				std::cerr<<"NBBLOCK invalid"<<std::endl;
		}
	}



	{
		char* str = getenv ("SUBSIZE");
		if (str) {
			std::stringstream ss (str);
			ss>>subsize;
			if (!ss)
				std::cerr<<"SUBSIZE invalid"<<std::endl;
		}
	}


	{
		char* str = getenv ("NBSTREAM");
		if (str) {
			std::stringstream ss (str);
			ss>>stream_number;
			if (!ss)
				std::cerr<<"NBSTREAM invalid"<<std::endl;
		}
	}

	if(nb_blocks == 0 && subsize == 0){
		std::cerr<<"SUBSIZE=??? or  NBBLOCK=???"<<std::endl;
		exit(0);
	}

	if(stream_number == 0){
		std::cerr<<"NBSTREAM=???? "<<std::endl;
		exit(0);
	}



	if(subsize == 0){
		int X;
		X = (int) nVtx/(nb_blocks) ;
		X = X / 32 ;
		subsize = (X+1) * 32;
	}


*/
	list<Task> *tasks = new  list<Task>;

	tbb::concurrent_bounded_queue<stream_container<int,int,float>* >* streams = new tbb::concurrent_bounded_queue<stream_container<int,int,float>* >;

	//
	bool coldcache = true;

	util::timestamp start(0,0);

	//cpuside variables  
	Scalar* prin_ = new Scalar[nVtx];
	EdgeType* xadj = xadj_;
	VertexType *adj = adj_;
	Scalar* val = val_;
	Scalar* prior = prior_;
	Scalar* prin = prin_;
	Scalar* prout = pr_;
	Scalar alpha = lambda;
	Scalar beta = 1-lambda;


	//cuda side variable
	EdgeType* d_xadj ;
	VertexType *d_adj ;
	Scalar* d_val ;
	Scalar* d_prior ;
	Scalar* d_prin ;
	Scalar* d_prout ;
	Scalar *d_alpha;
	Scalar *d_beta;


	//for test 
	Scalar* d_prout1 ;
	Scalar* d_prout2 ;


	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);

	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	//memalloc

	checkCudaErrors( cudaMalloc((void**)&d_xadj, (nVtx+1)*sizeof(*xadj)) );
	checkCudaErrors( cudaMalloc((void**)&d_adj, (xadj[nVtx])*sizeof(*adj)) );
	checkCudaErrors( cudaMalloc((void**)&d_val, (xadj[nVtx])*sizeof(*val)) );
	checkCudaErrors( cudaMalloc((void**)&d_prior, (nVtx*sizeof(*prior))));
	checkCudaErrors( cudaMalloc((void**)&d_prin, (nVtx*sizeof(*prin)) ));
	checkCudaErrors( cudaMalloc((void**)&d_prout1, (nVtx*sizeof(*prout)) ));
	checkCudaErrors( cudaMalloc((void**)&d_prout2, (nVtx*sizeof(*prout)) ));


	//cpu to gpu copies

	checkCudaErrors( cudaMemcpy(d_xadj, xadj, (nVtx+1)*sizeof(*xadj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_adj, adj, (xadj[nVtx])*sizeof(*adj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_val, val, (xadj[nVtx])*sizeof(*val), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_prior, prior, nVtx*sizeof(*prior), cudaMemcpyHostToDevice) );


	
	int nRows = nVtx;
	unsigned long* rowBlocks;
	const int nThreadPerBlock = 128; 
	const unsigned int blkSize = 1024; 
	const unsigned int blkMultiplier = 2 ;
	const unsigned int rows_for_vector = 1 ;
	const bool allocate_row_blocks = true;

	//device variable
	unsigned long* d_rowBlocks;
	unsigned int* d_blkSize;
	unsigned int* d_rows_for_vector;
	unsigned int* d_blkMultiplier;
	float* d_a;
	float* d_b;
	int rowBlockSize1;
	int rowBlockSize2;


	//calculer rowBlockSize
	rowBlockSize1 = ComputeRowBlocksSize<int,int>(xadj, nVtx, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock);
	cout << "rowBlockSize1 : " << rowBlockSize1 << endl;

	//declarer rowBlocks
	rowBlocks = (unsigned long*) calloc(sizeof(unsigned long),rowBlockSize1);

	//calculer rowBlocks
	ComputeRowBlocks<int,int>( rowBlocks, rowBlockSize2, xadj, nVtx, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock, allocate_row_blocks);
	cout << "rowBlockSize2 : " << rowBlockSize2 <<endl;




	int nb = split_input_to_tasks(rowBlocks, rowBlockSize1, 64, *tasks); 

	creat_stream(d_rowBlocks, d_a, d_b, d_val, d_xadj, d_adj, d_prin, d_prout, d_blkSize, d_rows_for_vector, d_blkMultiplier, streams, stream_number );

        cout << " ------------------" << stream_number << " --------------------- : " << endl;


	
	for(int i=0; i<stream_number; i++) {
		stream_container<int, int, float> *current_stream;
		Task t = get_task(tasks, i);
                streams->pop(current_stream);
                put_work_on_stream<int,int,float>(current_stream,t);
		
		cout << "stream id : " << current_stream->id <<
		" - d_rowBlocks - "<< current_stream->d_rowBlocks <<
		" - d_alpha - " << current_stream->alpha << 
                " - d_beta - " << current_stream->beta <<
		" - rowBlocksPtr - " << current_stream->rowBlocksPtr << 
		"- rowBlockSize - " << current_stream->rowBlockSize << endl;
	}
	

        cout << " ----- nb : " << nb << endl;

	

	

//	if(rowBlocks[rowBlockSize1] == 0){
//		rowBlockSize1--;
//	}

	int medium = cutRowBlocks(rowBlocks, rowBlockSize1);
 	int part2 = rowBlockSize1 - medium;
        cout << " - - medium : " << medium <<" - "<< rowBlocks[medium] << endl;
	

	for(int i=0; i<nb; i++){
		Task t = get_task(tasks, i);
		cout << "id : " << t.id <<" - rowBlocksPtr " << t.rowBlocksPtr <<" - rowBlockSize " << t.rowBlockSize <<endl;	

	}	
	for(int i=0; i<rowBlockSize1; i++){
		unsigned int row = ((rowBlocks[i] >> (64-32)) & ((1UL << 32) - 1UL));
		unsigned int row1 = (rowBlocks[i] & ((1UL << 24) - 1UL));
		cout << "  RowBlocks0["<< i <<"]   =  "<< row <<"  RowBlocks1["<< i <<"]   =  " << row1 << " |--> " << rowBlocks[i] <<  endl;
	}

/*
	//malloc for device variable
	checkCudaErrors( cudaMalloc((void**)&d_rowBlocks, (rowBlockSize1*sizeof(unsigned long))));
	checkCudaErrors( cudaMalloc((void**)&d_blkSize, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_rows_for_vector,1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_blkMultiplier, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_a, 1*sizeof(float)));
	checkCudaErrors( cudaMalloc((void**)&d_b, 1*sizeof(float)));


	//send data to device
	checkCudaErrors( cudaMemcpy(d_rowBlocks, rowBlocks, rowBlockSize1*sizeof(unsigned long), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkSize, &blkSize, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_rows_for_vector, &rows_for_vector, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkMultiplier, &blkMultiplier, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_a, &alpha, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_b, &beta, 1*sizeof(Scalar), cudaMemcpyHostToDevice) );

	int size =  (blkSize) * sizeof(float);

	csr_adaptative<<<(rowBlockSize1 + 1) , nThreadPerBlock, size >>>(d_val, d_adj, d_xadj, d_prior, d_prout1, d_rowBlocks, d_a,  d_b, d_blkSize, d_blkMultiplier, d_rows_for_vector, rowBlockSize1);


//	csr_adaptative<<<(medium + 1) , nThreadPerBlock, size >>>(d_val, d_adj, d_xadj, d_prior, d_prout1, d_rowBlocks, d_a,  d_b, d_blkSize, d_blkMultiplier, d_rows_for_vector, medium);


//	csr_adaptative<<<(part2 + 1) , nThreadPerBlock, size >>>(d_val, d_adj, d_xadj, d_prior, d_prout1, (d_rowBlocks + medium) , d_a,  d_b, d_blkSize, d_blkMultiplier, d_rows_for_vector, part2);

	 //(d_vals, d_cols, d_rowDelimiters, d_vec, d_out, d_rowBlocks, &d_a, &d_b, d_blkSize, d_blkMultiplier, d_rows_for_vector, rowBlockSize1);

	float* prout1 = (float*) malloc(nVtx*sizeof(float));
	float* prout2 = (float*) malloc(nVtx*sizeof(float));


      checkCudaErrors(cudaMemcpy(prout1, d_prout1, nVtx*sizeof(*prout), cudaMemcpyDeviceToHost));

	cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		nVtx, nVtx, xadj[nVtx], &alpha,
		descr,
		d_val, d_xadj, d_adj,
		d_prior, &beta,
		d_prout2);



	checkCudaErrors(cudaMemcpy(prout2, d_prout2, nVtx*sizeof(*prout), cudaMemcpyDeviceToHost));


	for(int i=0; i< nVtx; i++){
	
	//	if(prout2[i] != prout1[i])
		cout << i << " : " << prout1[i] << " - " << prout2[i] << " = " << (prout1[i] -prout2[i]) << endl;
	}

*/	
	
/*
	//	c_nst int* rowDelimiters = xadj ;
	//	int rowDelimiters1[] = {0,3,5,6,8,9} ;
	int rowDelimiters[] = {0,3,5,7,9,14} ;
	//	int cols1[] = {0,2,4,1,3,2,0,3,1};
	int cols[] = {0,2,4,1,3,2,4,0,3,0,1,2,3,4,5};
	//	float vals1[] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0};
	float vals[] = {1.0,2.0,3.0,4.0,5.0,6.0,1.0,7.0,8.0,1.0,9.0,1.0,1.0,1.0};
	float vec[] = {1.0,2.0,3.0,4.0,5.0};
	float out[] = {1.0,1.0,1.0,1.0,1.0};

	//	int rowDelimiters[] = {0,6,9,13,17,23,27,30,31} ;
	//	int cols[] = {0,2,4,5,6,7,2,4,7,0,2,4,5,1,3,5,7,0,2,3,4,5,6,0,2,4,6,1,3,5,3};
	//	float vals[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,};
	//	float vec[] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0};
	//	float out[] = {20.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

	int* d_rowDelimiters;
	int* d_cols;
	float* d_vals;
	float* d_vec;
	float* d_out;
	unsigned long* d_rowBlocks;
	float d_a = 0.5;
	float d_b = 0.5;

	const int nRows = 5;
	const int nThreadPerBlock = 4;
	const unsigned int blkSize = 3;
	const unsigned int blkMultiplier = 1 ;
	const unsigned int rows_for_vector = 1 ;

	const bool allocate_row_blocks = true;


	//device variable
	//unsigned long* d_rowBlocks;
	unsigned int* d_blkSize;
	unsigned int* d_rows_for_vector;
	unsigned int* d_blkMultiplier;


	int rowBlockSize1;
	int rowBlockSize2;

	//calculer rowBlockSize
	rowBlockSize1 = ComputeRowBlocksSize<int,int>(rowDelimiters, nRows, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock);

	cout << "rowBlockSize1 : " << rowBlockSize1 <<endl; 
	//declarer rowBlocks
	unsigned long* rowBlocks = (unsigned long*) calloc(sizeof(unsigned long),rowBlockSize1);

	//calculer rowBlocks
	ComputeRowBlocks<int,int>( rowBlocks, rowBlockSize2, rowDelimiters, nRows, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock, allocate_row_blocks);
	cout << "rowBlockSize2 : " << rowBlockSize2 <<endl; 


	if(rowBlocks[rowBlockSize1] == 0){
		rowBlockSize1--;
	}

	for(int i=0; i<rowBlockSize1; i++){
		unsigned int row = ((rowBlocks[i] >> (64-32)) & ((1UL << 32) - 1UL));
		unsigned int row1 = (rowBlocks[i] & ((1UL << 24) - 1UL));
		unsigned int wg = rowBlocks[i] & (1UL << 24);;
		cout << "  RowBlocks0["<< i <<"]   =  "<< row << " wg  " << wg <<"  RowBlocks1["<< i <<"]   =  " << row1 << " |--> " << rowBlocks[i] <<  endl;
	} 

	int *tab;

	checkCudaErrors( cudaMalloc((void**)&d_rowDelimiters, (nRows+1)*sizeof(int)) );
	checkCudaErrors( cudaMalloc((void**)&d_cols, (rowDelimiters[nRows])*sizeof(int)) );
	checkCudaErrors( cudaMalloc((void**)&d_vals, (rowDelimiters[nRows])*sizeof(float)) );
	checkCudaErrors( cudaMalloc((void**)&d_vec, (nRows*sizeof(float))));
	checkCudaErrors( cudaMalloc((void**)&d_out, (nRows*sizeof(float)) ));
	checkCudaErrors( cudaMalloc((void**)&d_rowBlocks, (rowBlockSize1*sizeof(unsigned long))));
	checkCudaErrors( cudaMalloc((void**)&tab, (rowBlockSize1*sizeof(int))));

	//malloc for device variable
	checkCudaErrors( cudaMalloc((void**)&d_blkSize, 1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_rows_for_vector,1*sizeof(unsigned int)));
	checkCudaErrors( cudaMalloc((void**)&d_blkMultiplier, 1*sizeof(unsigned int)));


	/send data to device
	checkCudaErrors( cudaMemcpy(d_blkSize, &blkSize, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_rows_for_vector, &rows_for_vector, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors( cudaMemcpy(d_blkMultiplier, &blkMultiplier, 1*sizeof(unsigned int), cudaMemcpyHostToDevice));


	checkCudaErrors( cudaMemcpy(d_rowDelimiters, rowDelimiters, (nRows+1)*sizeof(int), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_cols, cols, (rowDelimiters[nRows])*sizeof(int), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_vals, vals, (rowDelimiters[nRows])*sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_vec, vec, nRows*sizeof(float), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_out, out, nRows*sizeof(float), cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaMemcpy(d_rowBlocks, rowBlocks, rowBlockSize1*sizeof(unsigned long), cudaMemcpyHostToDevice));


	csr_adaptative<<<(rowBlockSize1+1), nThreadPerBlock>>>(d_vals, d_cols, d_rowDelimiters, d_vec, d_out, d_rowBlocks, &d_a, &d_b, d_blkSize, d_blkMultiplier, d_rows_for_vector, rowBlockSize1);


	int* out1 =  (int*)calloc(nRows,sizeof(int));
	float* out2 =  (float*)calloc(nRows,sizeof(float));

	cudaMemcpy(out2, d_out, nRows*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(out1, tab, rowBlockSize1*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i< rowBlockSize1; i++) 
		cout << i << " : " << out1[i]  <<endl;

	for(int i=0; i< nRows; i++) 
		cout << "d_out [" << i << "] : " << out2[i]  <<endl;

*/
	//	*{	int* d_c;
	/*	  int* c;

		  cudaMalloc((void**)&d_c, 10*sizeof(int)) ;
		  c = (int *) malloc(10 * sizeof(int));	

		  csr_adaptativeT<<<2,10>>>(d_c);

		  cudaMemcpy(c, d_c, 10*sizeof(int), cudaMemcpyDeviceToHost);	

		  for(int i=0; i< 10; i++) 
		  cout << i << " : " << c[i]  <<endl;

		  rowBlockSize1 =	ComputeRowBlocksSize<int,int>(rowDelimiters,nRows,blkSize,blkMultiplier,rows_for_vector);

		  cout  << "--------------- nRows " << rowBlockSize1 << std::endl;

		  unsigned long * rowBlocks = (unsigned long*) calloc(sizeof(unsigned long),rowBlockSize1);


		  ComputeRowBlocks<int,int>( rowBlocks, rowBlockSize2, rowDelimiters, nRows, blkSize, blkMultiplier, rows_for_vector, allocate_row_blocks );

		  cout  << "--------------- nRows " << rowBlockSize2 << std::endl;

		  unsigned int v = 0 ;
		  for(int i=0; i<rowBlockSize1; i++){
		  unsigned int row = ((rowBlocks[i] >> (64-32)) & ((1UL << 32) - 1UL));
		  unsigned int row1 = (rowBlocks[i] & ((1UL << 24) - 1UL));
		  unsigned int wg = rowBlocks[i] & (1UL << 24);;
		  if(row == v) 
		  cout << "------- long row khatiri" << endl;
		  cout << "  RowBlocks0["<< i <<"]   =  "<< row << " wg  " << wg <<"  RowBlocks1["<< i <<"]   =  " << row1 << " |--> " << rowBlocks[i] <<  endl;

		  } 

	 */




	return 0;
}



