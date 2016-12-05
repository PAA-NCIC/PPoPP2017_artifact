All:sgemm sgemm_tt_128x128_template.cubin sgemm_tt_128x128.cubin

sgemm:sgemm.cpp
	nvcc -o $@ $<  -lcudart -lcuda -lcublas
sgemm_tt_128x128_template.cubin:sgemm_tt_128x128.cu
	nvcc -arch sm_35 -m 64 $<  -cubin -o $@

sgemm_tt_128x128.cubin: sgemm_tt_128x128.sass 
	KeplerAs.pl -i $< sgemm_tt_128x128_template.cubin $@
clean:
	rm sgemm sgemm_tt_128x128_template.cubin sgemm_tt_128x128.cubin
