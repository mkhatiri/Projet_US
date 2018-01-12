#! /bin/bash




#NPS -N mkhatiri

#PBS -q python

#PBS -l nodes=1:ppn=1:gpus=3 


module load cuda

export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

echo "exec"
cd /users/mkhatiri/projet/src/page_rank 


#lspci -t

#nvidia-smi topo -m

#nvidia-smi -a

for i in $(ls /users/mkhatiri/PROJETCOMPLET/SNAP/*.bin) 
do
#NBBLOCK=256 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative $i 
#NBSTREAM=4 NBBLOCK=16 ./new-cuda-pr-2gpus $i;
#NBBLOCK=256 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative-2gpus $i;
#NBSTREAM=4 NBBLOCK=16 ./new-cuda-pr-2gpus $i;

NBBLOCK=128 NBTHREAD=256 BLKSIZE=2048 ./new-cuda-adaptative $i 
NBSTREAM=2 NBBLOCK=32 ./new-cuda-pr $i

done
#./p 100 10 4 1 0 #

#NBBLOCK=256 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin


#echo "64 512 2048"
#NBBLOCK=64 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin
#echo "128 512 2048"
#NBBLOCK=128 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin
#echo "256 512 2048"
#NBBLOCK=256 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin
#echo "512 512 2048"
#NBBLOCK=256 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/soc-LiveJournal1.bin


#NBBLOCK=128 NBTHREAD=256 BLKSIZE=2048 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/web-Google.bin
#NBSTREAM=2 NBBLOCK=32 ./new-cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/web-Google.bin

#NBBLOCK=128 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0312.bin

#NBBLOCK=256 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0312.bin

#NBBLOCK=512 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0312.bin


#NBBLOCK=256 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0312.bin


#NBSTREAM=2 NBBLOCK=4 ./new-cuda-lightspmv-pr /users/mkhatiri/PROJETCOMPLET/SNAP/soc-LiveJournal1.bin


