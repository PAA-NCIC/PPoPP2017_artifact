#!/bin/bash
#1k2k4k
file="1k2k4k.txt"
for ((i = 384; i <= 3072; i = i + 384))
do

    echo "matrix: " $i
    M=$i
    K=$(($i * 2))
    N=$(($i * 4))
    ./sgemm $M $K $N  >> $file
    sleep 5
done


#1k4kk
file="1k4k1k.txt"
for ((i = 384; i <= 3072; i = i + 384))
do

    echo "matrix: " $i
    M=$(($i * 1))
    K=$(($i * 4))
    N=$(($i * 1))
    ./sgemm $M $K $N  >> $file
    sleep 5
done

#4k2kk
file="4k2k1k.txt"
for ((i = 384; i <= 3072; i = i + 384))
do

    echo "matrix: " $i
    M=$(($i * 4))
    K=$(($i * 2))
    N=$(($i * 1))
    ./sgemm $M $K $N  >> $file
    sleep 5
done
#4k1k4k
file="4k1k4k.txt"
for ((i = 384; i <= 3072; i = i + 384))
do

    echo "matrix: " $i
    M=$(($i * 4))
    K=$(($i * 1))
    N=$(($i * 4))
    ./sgemm $M $K $N  >> $file
    sleep 5
done
