DPCC = icpx
CC = g++
DPCCFLAGS = -fsycl -O3
CCFLAGS = -O3

all: linear_sliding parallel_sliding linear_sliding_multiclass parallel_sliding_multiclass

linear_sliding: main.o util.o linear_sliding.o
	$(CC) $(CFLAGS) -o linear_sliding main.o util.o linear_sliding.o

linear_sliding_multiclass: main.o util.o linear_sliding_multiclass.o
	$(CC) $(CFLAGS) -o linear_sliding_multiclass main.o util.o linear_sliding_multiclass.o

parallel_sliding: main.o util.o parallel_sliding.o
	$(DPCC) $(DPCCFLAGS) -o parallel_sliding main.o util.o parallel_sliding.o

parallel_sliding_multiclass: main.o util.o parallel_sliding_multiclass.o
	$(DPCC) $(DPCCFLAGS) -o parallel_sliding_multiclass main.o util.o parallel_sliding_multiclass.o

linear_base: main.o util.o linear_base.o
	$(CC) $(CFLAGS) -o linear_base main.o util.o linear_base.o

parallel_base_internal: main.o util.o parallel_base_internal.o
	$(DPCC) $(DPCCFLAGS) -o parallel_base_internal main.o util.o parallel_base_internal.o

main.o: main.cc util.h
	$(CC) $(CCFLAGS) -c main.cc

util.o: util.cc util.h
	$(CC) $(CCFLAGS) -c util.cc

linear_sliding.o: linear_sliding.cc
	$(CC) $(CCFLAGS) -c linear_sliding.cc

linear_sliding_multiclass.o: linear_sliding_multiclass.cc
	$(CC) $(CCFLAGS) -c linear_sliding_multiclass.cc

parallel_sliding.o: parallel_sliding.cc
	$(DPCC) $(DPCCFLAGS) -c parallel_sliding.cc

parallel_sliding_multiclass.o: parallel_sliding_multiclass.cc
	$(DPCC) $(DPCCFLAGS) -c parallel_sliding_multiclass.cc

linear_base.o: linear_base.cc 
	$(CC) $(CCFLAGS) -c linear_base.cc

parallel_base_internal.o: parallel_base_internal.cc
	$(DPCC) $(DPCCFLAGS) -c parallel_base_internal.cc

clean: 
	rm -f *.o

clear:
	rm -f *.o linear_sliding parallel_sliding linear_sliding_multiclass parallel_sliding_multiclass
