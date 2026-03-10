CC := gcc
BINDIR := bin
OBJDIR := .obj

COMMON_FLAGS := -Wall -Wextra -DROOT_DIR=\"$(shell pwd)\" -Iinc -Isrc

ifeq ($(DEBUG), 1)
MODE_FLAGS := -Og -g -fsanitize=address
TAG := debug
else
MODE_FLAGS := -O3
TAG := release
endif

all: build

build: clean dirs
	@echo "Building no-omp"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -DNO_OMP -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -DNO_OMP -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -DNO_OMP -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -DNO_OMP -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -DNO_OMP $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/no-omp -lm

	@echo "Building intra-layer"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/intra-layer -lm

	@echo "Building inter-sample"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/inter-sample -lm

	@echo "Building inter-sample-vectorized"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/inter-sample-vectorized -lm

dirs:
	@mkdir -p $(BINDIR)/$(TAG)
	@mkdir -p $(OBJDIR)

clean:
	@rm -rf $(OBJDIR) $(BINDIR)

fftest: clean dirs
	@echo "Building feedforward critical section test"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_CRITICAL -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_CRITICAL -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_CRITICAL -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_CRITICAL -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_CRITICAL $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/ff-critical -lm

	@echo "Building feedforward inner loop test"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_INNER_LOOP -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_INNER_LOOP -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_INNER_LOOP -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_INNER_LOOP -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_INNER_LOOP $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/ff-inner-loop -lm

	@echo "Building feedforward outer loop test"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_OUTER_LOOP -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_OUTER_LOOP -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_OUTER_LOOP -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_OUTER_LOOP -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_OUTER_LOOP $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/ff-outer-loop -lm

	@echo "Building feedforward simd test"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_SIMD -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_SIMD -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_SIMD -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_SIMD -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DTEST_SIMD $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/ff-simd -lm
