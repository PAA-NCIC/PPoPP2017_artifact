All:sgemm

sgemm:sgemm.cpp
	nvcc -o $@ $<  -lcudart -lcuda -lcublas
clean:
	rm sgemm

