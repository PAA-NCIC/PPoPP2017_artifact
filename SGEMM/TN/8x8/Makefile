All:sgemm sgemm_tn_128x128_template.cubin sgemm_tn_128x128.cubin

sgemm:sgemm.cpp
	nvcc -o $@ $<  -lcudart -lcuda -lcublas
sgemm_tn_128x128_template.cubin:sgemm_tn_128x128.cu
	nvcc -arch sm_35 -m 64 $<  -cubin -o $@

sgemm_tn_128x128.cubin: sgemm_tn_128x128.sass 
	KeplerAs.pl -i $< sgemm_tn_128x128_template.cubin $@
clean:
	rm sgemm sgemm_tn_128x128_template.cubin sgemm_tn_128x128.cubin
