# make
matmul: src/main.cpp src/matmul.cpp
	g++ src/main.cpp src/matmul.cpp -o bin/matmul -O3 -pthread -fopenmp

all: matmul

clean:
	rm -f bin/matmul