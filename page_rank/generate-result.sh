#!/bin/sh

number_blocks="4 8 16 32 64"
number_stream="2 4"
program="new-cuda-lightspmv-pr-2gpus"


for p in $program
do
	echo  "			----------- $p-------------------- " 
	echo  "			----------- $p-------------------- " >> $p.log
	for ns in $number_stream
	do
		echo "------- streams $ns ---------- " 
		echo "			------- $nb ---------- " >> $p.log
		for nb in $number_blocks 
		do

			echo "------- blocks $nb ---------- " 
			echo "			------- $nb ---------- " >> $p.log

			for file in `find /home/esaule1/Graphs/SNAP/ -iname '*.bin'`
			do
				NBSTREAM=$ns NBBLOCK=$nb ./$p $file 10 2>/dev/null >> $p.log
				echo  " number-blocks: $nb " >> $p.log
			done
		done
	done	
done
