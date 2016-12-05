## Benchmarking SGEMM of cuBLAS and our optimized SGEMM
* Clean the binaries
    * $make clean
* Compile the code
    * $ make
* remove the result file if it exists, otherwise nextstep
    * $ rm ppopp_time.txt
* Run the code
    * $ ./sgemm.sh
* Observe the result
    * $ vim ppopp_time.txt
