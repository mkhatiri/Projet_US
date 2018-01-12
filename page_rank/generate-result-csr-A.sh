#!/bin/sh

nb_thread="256 512 1024"

blk_size="1024 2048 4096"

nb_block="2 4 8 32 64 128 256"

program="new-cuda-adaptative new-cuda-adaptative-2gpus"

err="err"


for p in $program
do
	echo  "----------- $p-------------------- " 
	
 	for nbT in $nb_thread
	do
		echo "------- nb thread per bloc $nbT ---------- " 
		echo  "----------- $p  - $nbT thread per blocks -------------------- " >> $p.$nbT.log 
		echo  "----------- $p  - $nbT thread per blocks -------------------- " >> $err.log 
		
		for bks in $blk_size 
		do
			echo "------- blocks Size $bks ---------- " 
			echo "			------- $bks ---------- " >> $p.$nbT.log
			echo "			------- $bks ---------- " >> $err.log
			for nbblock in $nb_block
                        do

				echo "------- number blocks  $nbblock ---------- " 
				echo "------- number blocks  $nbblock ---------- " >> $p.$nbT.log 
				echo "------- number blocks  $nbblock ---------- " >> $err.log 
				for file in `find /home/esaule1/Graphs/SNAP/ -iname '*.bin'`
				do
				NBBLOCK=$nbblock NBTHREAD=$nbT BLKSIZE=$bks ./$p $file 10 2>>$err.log >> $p.$nbT.log
					echo  " blocks_Size: $bks nb_thread: $nbT " >> $p.log
				done
			done
		done
	done	
done
