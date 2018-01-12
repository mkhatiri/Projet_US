

CXXFLAGS += 

MKL_LDFLAGS += 

INCLUDES   = -I./

TARGET     =  omp-pr

CXXFILES   = omp-pr

SRCS	= $(CFILES:%=%.c) $(CXXFILES:%=%.cpp)

TBBLDFLAGS= -L /opt/intel/composerxe/tbb/lib/mic -ltbb 


all: $(TARGET)


mpi2-omp-pr: mpi2-omp-pr.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o mpi2-omp-pr.o $(LIBS) 

mpiN-omp-pr: mpiN-omp-pr.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o mpiN-omp-pr.o $(LIBS) 


omp-pr: omp-pr.o graphIO.o logged_array.o
	$(LD) $(LDFLAGS) -o $@ graphIO.o omp-pr.o logged_array.o $(LIBS) 

cachecount: cachecount.o graphIO.o logged_array.o
	$(LD) $(LDFLAGS) -o $@ graphIO.o cachecount.o logged_array.o $(LIBS) 


cuda-pr: cuda-pr.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o cuda-pr.o $(LIBS) $(CUDALDFLAGS)

cuda-lightspmv-pr: cuda-lightspmv-pr.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o cuda-lightspmv-pr.o $(LIBS) $(CUDALDFLAGS)

cuda-lightspmv-pr-2gpus: cuda-lightspmv-pr-2gpus.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o cuda-lightspmv-pr-2gpus.o $(LIBS) $(CUDALDFLAGS)


cuda-pr-2gpus: cuda-pr-2gpus.o graphIO.o 
	$(LD) $(LDFLAGS) -o $@ graphIO.o cuda-pr-2gpus.o $(LIBS) $(CUDALDFLAGS)


include makefile.in
# DO NOT DELETE

omp-pr.o: main-pr.hpp timestamp.hpp graphIO.hpp Padded2DArray.hpp
omp-pr.o: memutils.hpp logged_array.hpp Model.hpp
