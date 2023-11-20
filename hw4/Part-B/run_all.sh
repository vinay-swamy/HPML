#!/bin/bash 
time_results="time_results.txt"
bash run_q1.sh > $time_results
bash run_q2.sh 1 1  >> $time_results
bash run_q2.sh 1 256 >> $time_results
bash run_q2.sh -1 256 >> $time_results
bash run_q3.sh 1 1  >> $time_results
bash run_q3.sh 1 256 >> $time_results
bash run_q3.sh -1 256 >> $time_results