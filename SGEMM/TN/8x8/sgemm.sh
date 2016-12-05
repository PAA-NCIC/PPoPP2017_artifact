#!/bin/bash
echo "benchmarking our 8x8 blocking SGEMM and cuBLAS SGEMM ..."

file="ppopp_time.txt";
echo "our SGEMM and cuBLAS SGEMM peformance is writen in $file"
for ((i = 768; i <= 12288; i = i + i))
do
    echo "Matrix dimension: [M, K, N] " $i, $i, $i
    ./sgemm $i $i $i >> $file
    sleep 5
done
