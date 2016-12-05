## Benchmarking SGEMM of cuBLAS and our optimized SGEMM
* Clean the binaries
    * $make clean
* remove the result file if it exists
    * $ rm ppopp_time.txt
* Compile the code
    * $ make
* Run the code
    * $ ./sgemm.sh
*Observe the result
    * $ vim ppopp_time.txt
