CUDAARCH ?= sm_35

ptx:	ptxgen
		mkdir -p data/ptx
		./ptxgen -a=$(CUDAARCH)
clean:
	- rm -rf  data
