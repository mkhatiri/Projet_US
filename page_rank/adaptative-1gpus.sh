#! /bin/bash




#NPS -N mkhatiri

#PBS -q python

#PBS -l nodes=1:ppn=2:gpus=3 


module load cuda/8.0

echo "exec"
cd /users/mkhatiri/projet/src/page_rank 

#lspci -t

#nvidia-smi topo -m

#nvidia-smi -a

#NBBLOCK=128 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/web-Google.bin
NBBLOCK=128 NBTHREAD=128 BLKSIZE=1024 ./new-cuda-adaptative  /users/mkhatiri/PROJETCOMPLET/SNAP/web-Google.bin

