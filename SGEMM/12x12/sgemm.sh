#!/bin/bash

##compile code ###
echo "...............compiling the assembly code .........."
echo "....................................................."
KeplerAs.pl -i sgemm_tn_192x192.sass sgemm_tn_192x192_template.cubin sgemm_tn_192x192.cubin
echo "............... compile sgemm.cpp ..................."
echo "....................................................."
make

file="ppopp_time.txt";
echo "our SGEMM and cuBLAS SGEMM peformance is writen in $file"
for ((i = 768; i <= 12288; i = i + i))
do
    echo "Matrix dimension: [M, K, N] " $i, $i, $i
    ./sgemm $i $i $i >> $file
done
