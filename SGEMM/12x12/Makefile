All:sgemm sgemm_tn_192x192_template.cubin sgemm_tn_192x192.cubin

sgemm:sgemm.cpp
	nvcc -o $@ $<  -lcudart -lcuda -lcublas
sgemm_tn_192x192_template.cubin:sgemm_tn_192x192.cu
	nvcc -arch sm_35 -m 64 $<  -cubin -o $@

sgemm_tn_192x192.cubin: sgemm_tn_192x192.sass 
	KeplerAs.pl -i $< sgemm_tn_192x192_template.cubin $@
clean:
	rm sgemm sgemm_tn_192x192_template.cubin sgemm_tn_192x192.cubin
