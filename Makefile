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
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -DNO_OMP -c src/mlp.c -o $(OBJDIR)/mlp.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -DNO_OMP -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -DNO_OMP $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/no-omp -lm

	@echo "Building intra-layer"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/mlp.c -o $(OBJDIR)/mlp.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTRA_LAYER $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/intra-layer -lm

	@echo "Building inter-sample"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/mlp.c -o $(OBJDIR)/mlp.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DINTER_SAMPLE $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/inter-sample -lm

	@echo "Building inter-sample-vectorized"
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/profiler.c -o $(OBJDIR)/profiler.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/mlp.c -o $(OBJDIR)/mlp.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE -c src/main.c -o $(OBJDIR)/main.o
	$(CC) $(COMMON_FLAGS) $(MODE_FLAGS) -fopenmp -DVECTORIZED -DINTER_SAMPLE $(OBJDIR)/*.o -o $(BINDIR)/$(TAG)/inter-sample-vectorized -lm

dirs:
	@mkdir -p $(BINDIR)/$(TAG)
	@mkdir -p $(OBJDIR)

clean:
	@rm -rf $(OBJDIR) $(BINDIR)
