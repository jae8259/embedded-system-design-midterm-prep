NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17 -arch=sm_87 -Xcompiler -fopenmp
PREFIX_SUM_SRC ?= src/prefix_sum.cu
PREFIX_SUM_LANGFLAGS ?=
TARGET ?= bin/prob2

prob4: src/main.cpp $(PREFIX_SUM_SRC)
	mkdir -p $(dir $(TARGET))
	$(NVCC) $(NVCCFLAGS) src/main.cpp $(PREFIX_SUM_LANGFLAGS) $(PREFIX_SUM_SRC) -o $(TARGET)

all: prob4

clean:
	rm -f $(TARGET)
