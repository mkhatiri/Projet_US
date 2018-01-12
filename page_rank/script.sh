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

for i in $(ls /users/mkhatiri/PROJETCOMPLET/SNAP/*.bin) 
do
echo $i;
done;

