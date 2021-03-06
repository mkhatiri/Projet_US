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

	int nb_blocks = 0;
	int blk_size = 0;
	int nb_threads = 0;

	{
		char* str = getenv ("NBTHREAD");
		if (str) {
			std::stringstream ss (str);
			ss>>nb_threads;
			if (!ss)
				std::cerr<<"NBTHREAD invalid"<<std::endl;
		}
	}



	{
		char* str = getenv ("BLKSIZE");
		if (str) {
			std::stringstream ss (str);
			ss>>blk_size;
			if (!ss)
				std::cerr<<"SUBSIZE invalid"<<std::endl;
		}
	}


	{
                char* str = getenv ("NBBLOCK");
                if (str) {
                        std::stringstream ss (str);
                        ss>>nb_blocks;
                        if (!ss)
                                std::cerr<<"NBBLOCK invalid"<<std::endl;
                }
        }



	if(nb_threads == 0 ){
		std::cerr<<" NBTHREAD=??? "<<std::endl;
		exit(0);
	}

	if(blk_size == 0 ){
                std::cerr<<" BLKSIZE=??? "<<std::endl;
                exit(0);
        }
	if(nb_blocks == 0 ){
                std::cerr<<" NBBLOCK=??? "<<std::endl;
                exit(0);
        }

	

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
	checkCudaErrors( cudaMalloc((void**)&d_prout, (nVtx*sizeof(*prout)) ));

	//cpu to gpu copies

	checkCudaErrors( cudaMemcpy(d_xadj, xadj, (nVtx+1)*sizeof(*xadj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_adj, adj, (xadj[nVtx])*sizeof(*adj), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_val, val, (xadj[nVtx])*sizeof(*val), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(d_prior, prior, nVtx*sizeof(*prior), cudaMemcpyHostToDevice) );



	int nRows = nVtx;
	unsigned long* rowBlocks;
	const int nThreadPerBlock = nb_threads; 
	const unsigned int blkSize = blk_size;
	const unsigned int blkMultiplier = 3 ;
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

	//declarer rowBlocks
	rowBlocks = (unsigned long*) calloc(sizeof(unsigned long),rowBlockSize1);

	cout << " rowBlockSize1 - 1 " << rowBlockSize1 << endl;
	//calculer rowBlocks
	ComputeRowBlocks<int,int>( rowBlocks, rowBlockSize1, xadj, nVtx, blkSize, blkMultiplier, rows_for_vector, nThreadPerBlock, allocate_row_blocks);

	cout << " rowBlockSize1 - 2 " << rowBlockSize1 << endl;


	//	if(rowBlocks[rowBlockSize1] == 0){
	//		rowBlockSize1--;
	//	}


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


	int mmshared_size =  (blkSize + 1) * sizeof(float);


	// prepar stream config

	list<Task> *tasks = new  list<Task>;

	tbb::concurrent_bounded_queue<stream_container<int,int,float>* >* streams = new tbb::concurrent_bounded_queue<stream_container<int,int,float>* >;


//	int nb_blocks = 128;
	int stream_number = 2;


	int X, subsize;
	X = (int) rowBlockSize1/(nb_blocks) ;

	if(X % 64 == 0){
		subsize = X;  
	}else{
		X = X / 64 ;
		subsize = (X+1) * 64;
	}


	rowBlockSize1--;

	int xadjPtr1 =  ((rowBlocks[rowBlockSize1] >> (64-32)) & ((1UL << 32) - 1UL));

	int xadjPtr2;
	for(int i=0; i<=rowBlockSize1; i++)
		{	
			xadjPtr2 = ((rowBlocks[i] >> (64-32)) & ((1UL << 32) - 1UL));
			cout << "xadjPtr1="  << xadjPtr2 << endl ;
		}
//	cout << " rowBlocks[rowBlockSize1]  " << rowBlocks[rowBlockSize1] << "rowBlockSize1 : "<< rowBlockSize1 << "last row " << xadjPtr1 << endl;
	
//	cout << "subsize : "<< subsize << endl;
//	cout << "start creat stream " <<endl;


	creat_stream<int, int, float>(d_rowBlocks, d_a, d_b, d_val, d_xadj, d_adj, d_prin, d_prout, d_blkSize, d_rows_for_vector, d_blkMultiplier, streams, stream_number );
	cout << "end creat stream " <<endl;

	cout << "start split task " <<endl;
	cout << " rowBlocks=" << rowBlocks << " rowBlockSize1=" << rowBlockSize1 << "subsize" << subsize <<endl;
	int nb_tasks = split_input_to_tasks(rowBlocks, rowBlockSize1, subsize, *tasks);
	cout << "fin split task " <<endl;


	cout << "nb_tasks " << nb_tasks << endl;


/*
	for(int i=0; i<nb_tasks; i++){
		Task t = get_task(tasks, i);
		cout << "id : " << t.id <<" - rowBlocksPtr " << t.rowBlocksPtr <<" - rowBlockSize " << t.rowBlockSize <<endl;

	}
*/



	for (int TRY=0; TRY<THROW_AWAY+nTry; ++TRY)
	{
		if (TRY >= THROW_AWAY)
			start = util::timestamp();

		for (int iter = 0; iter < 40 ; ++iter) {

			int index = 0;
			//setup prin
			if (iter == 0)
				//std::copy (prior, prior+nVtx, prin);
				checkCudaErrors(cudaMemcpy(d_prin, d_prior, nVtx*sizeof(*prior), cudaMemcpyDeviceToDevice));
			else
				//std::copy (prout, prout+nVtx, prin);
				checkCudaErrors(cudaMemcpy(d_prin, d_prout, nVtx*sizeof(*prout), cudaMemcpyDeviceToDevice));


			cudaPrintError("after if");
			Scalar eps = 0.;

			//prout = A prin
			//prout = lambda * prout + (1-lambda) prior

			checkCudaErrors(cudaMemcpy(d_prout, d_prior, nVtx*sizeof(*prior), cudaMemcpyDeviceToDevice));


			//for float it is S.
			//does prout = alpha A prin + beta prout
	//		cerr << "nb_task=" << nb_tasks <<endl;

			while(index < nb_tasks-1){
				stream_container<int, int, float> *current_stream;
				Task t = get_task(tasks, index++);
				streams->pop(current_stream);
				put_work_on_stream<int,int,float>(current_stream,t);
				//cudaPrintError("before kernel");


		
			//	cout << " current_stream->rowBlockSize "  << current_stream->rowBlockSize <<endl;
			//	cout <<" current_stream->rowBlocksPtr "<<  current_stream->rowBlocksPtr <<endl;
			//	cout <<" current_stream->d_rowBlocks "<< current_stream->d_rowBlocks <<endl;
			//	cout <<" current_stream->stream "<< current_stream->stream  <<endl;

				


				csr_adaptative<<<current_stream->rowBlockSize , nThreadPerBlock, mmshared_size ,current_stream->stream>>>(current_stream->d_val, current_stream->d_adj, current_stream->d_xadj, current_stream->d_prin, current_stream->d_prout, (current_stream->d_rowBlocks + current_stream->rowBlocksPtr ), current_stream->alpha, current_stream->beta, current_stream->d_blkSize, current_stream->d_blkMultiplier, current_stream->d_rows_for_vector, current_stream->rowBlockSize);


                                cudaPrintError("befor callback");
				cudaStreamAddCallback(current_stream->stream, call_back , current_stream , 0);
                                cudaPrintError("after callback");

			checkCudaErrors(cudaMemcpy(prout, current_stream->d_prout, 10*sizeof(*prout), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(prin, current_stream->d_prin, 10*sizeof(*prin), cudaMemcpyDeviceToHost));
		//	std::cerr<< index  << "PR="<<prout[3]<< " - "<< prin[3] << std::endl; 

			}

			cudaThreadSynchronize();


			//compute epsilon
			//using prin to compute epsilon
			float epsalpha = -1.0f;
			cublasStatus = cublasSaxpy (cublasHandle, nVtx, &epsalpha, d_prout, 1, d_prin, 1); // d_prin = d_prout*-1 + d_prin
			cudaPrintError("after cublas");

			if (cublasStatus != CUBLAS_STATUS_SUCCESS)
				std::cerr<<"err_2"<<std::endl;

			cublasStatus = cublasSasum(cublasHandle, nVtx, d_prin, 1, &eps);
			if (cublasStatus != CUBLAS_STATUS_SUCCESS)
				std::cerr<<"err_3"<<std::endl;


			//stopping condition
			if (eps < 0) // deactivited for testing purposes
				iter = 20;

//			std::cerr<<eps<<std::endl;

		}

		checkCudaErrors(cudaMemcpy(prout, d_prout, nVtx*sizeof(*prout), cudaMemcpyDeviceToHost));

	
		std::cerr <<" PR="<<prout[0]<< " " << prout[1] << " " << prout[2] << std::endl; 

		if (TRY >= THROW_AWAY)
		{
			util::timestamp stop;  
			totaltime += stop - start;
		}



		/*    
#ifndef LOG
if (coldcache) {
#pragma omp parallel
{
evict_array_from_cache(adj, xadj[nVtx]*sizeof(*adj));
evict_array_from_cache(xadj, (nVtx+1)*sizeof(*xadj));
evict_array_from_cache(val, xadj[nVtx]*sizeof(*val));
evict_array_from_cache(prior, nVtx*sizeof(*prior));
evict_array_from_cache(prin, nVtx*sizeof(*prin));
evict_array_from_cache(prout, nVtx*sizeof(*prout));

#pragma omp barrier
}
}
#endif
		 */
}


cudaFree(d_rowBlocks);
cudaFree(d_blkSize);
cudaFree(d_rows_for_vector);
cudaFree(d_blkMultiplier);
cudaFree(d_a);
cudaFree(d_b);



#ifdef SHOWLOADBALANCE
std::cout<<"load balance"<<std::endl;
for (int i=0; i< 244; ++i)
std::cout<<count[i]<<std::endl;
#endif

delete[] prin_;



return 0;
}



