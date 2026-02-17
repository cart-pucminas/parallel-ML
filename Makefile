COMMON_FLAGS := -Wall -Wextra -fopenmp -DROOT_DIR=\"$(shell pwd)\"
OPTMIZATION_FLAGS := 

vectorized: OPTMIZATION_FLAGS := -DVECTORIZED
parallelin: OPTMIZATION_FLAGS := $(OPTMIZATION_FLAGS) -DINTRA_LAYER
parallelout: OPTMIZATION_FLAGS := $(OPTMIZATION_FLAGS) -DINTER_SAMPLE

CFLAGS := $(COMMON_FLAGS) $(OPTMIZATION_FLAGS) -O3
EXEFOLDER := release
OBJDIR := .obj
BINDIR := bin

debug: CFLAGS := $(COMMON_FLAGS) -Og -g -fsanitize=address
debug: EXEFOLDER := debug

all: build

debug: build

build: dirs objs exe

dirs:
	mkdir -p $(BINDIR)/$(EXEFOLDER)
	mkdir -p $(OBJDIR)

objs:
	gcc $(CFLAGS) -fopenmp -Iinc -Isrc -c src/profiler.c -o $(OBJDIR)/profiler.o
	gcc $(CFLAGS) -fopenmp -Iinc -Isrc -c src/mnist_dataloader.c -o $(OBJDIR)/mnist_dataloader.o
	gcc $(CFLAGS) -fopenmp -Iinc -Isrc -c src/xor_dataloader.c -o $(OBJDIR)/xor_dataloader.o
	gcc $(CFLAGS) -fopenmp -Iinc -Isrc -c src/mlp.c -o $(OBJDIR)/mlp.o
	gcc $(CFLAGS) -fopenmp -Iinc -Isrc -c src/main.c -o $(OBJDIR)/main.o

exe:
	gcc $(CFLAGS) -fopenmp $(OBJDIR)/*.o -o $(BINDIR)/$(EXEFOLDER)/main -lm

clean:
	rm -fr $(OBJDIR) $(BINDIR)
