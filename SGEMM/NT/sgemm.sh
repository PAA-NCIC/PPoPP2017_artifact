#!/bin/bash

##compile code ###
make
file="ppopp_time.txt";
echo "our SGEMM and cuBLAS SGEMM peformance is writen in $file"
for ((i = 768; i <= 768; i = i + i))
do
    echo "Matrix dimension: [M, K, N] " $i, $i, $i
    ./sgemm $i $i $i >> $file
    sleep 5
done
