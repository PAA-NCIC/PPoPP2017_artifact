All:sgemm sgemm_nn_128x128_template.cubin sgemm_nn_128x128.cubin

sgemm:sgemm.cpp
	nvcc -o $@ $<  -lcudart -lcuda -lcublas
sgemm_nn_128x128_template.cubin:sgemm_nn_128x128.cu
	nvcc -arch sm_35 -m 64 $<  -cubin -o $@

sgemm_nn_128x128.cubin: sgemm_nn_128x128.sass 
	KeplerAs.pl -i $< sgemm_nn_128x128_template.cubin $@
clean:
	rm sgemm sgemm_nn_128x128_template.cubin sgemm_nn_128x128.cubin
