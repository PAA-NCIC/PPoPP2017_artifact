#!/bin/bash

file="ppopp_time.txt";
echo "our SGEMM and cuBLAS SGEMM peformance is writen in $file"
for ((i = 512; i <= 4096; i = i + 512))
do
    echo "Matrix dimension: [M, K, N] " $i, $i, $i
    ./sgemm $i $i $i >> $file
    sleep 5
done
