#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../support/matmul.h"
#include "../support/timer.h"

int main(int argc, char * argv[]){

    //read matrix from given input files
    FILE* input_mat = fopen(argv[1], "r");
    FILE* output_mat = fopen(argv[2], "r");

    if(!input_mat) {
		fprintf(stderr, "Failed to open %s: ", argv[1]);
        perror("Failed to open");
		return 0;
    }

    int m, n, k;

    fscanf(input_mat, "%d %d\n", &m, &k);
    fscanf(output_mat, "%d\n", &n);

    printf("Input size - M: %d, N: %d, K: %d\n", m, n, k);

    //allocate memory for the matrices
    int *input_A = (int *)calloc(m*k, sizeof(int));
    int *input_B = (int *)calloc(k*n, sizeof(int));
    int *output = (int *)calloc(m*n, sizeof(int));
    int *output_ref = (int *)calloc(m*n, sizeof(int));

    //Read the matrices into allocated memory
    for(int x=0; x<m; x++){
        for(int y=0; y<k; y++){
            fscanf(input_mat, "%d", input_A + (x * k + y));
        }
    }
    for(int x=0; x<k; x++){
        for(int y=0; y<n; y++){
            fscanf(input_mat, "%d", input_B + (x * n + y));
        }
    }
    for(int x=0; x<m; x++){
        for(int y=0; y<n; y++){
            fscanf(output_mat, "%d", output_ref + (x * n + y));
        }
    }
    fclose(input_mat);
    fclose(output_mat);

    Timer time;

    //compute matrix multiplication
    start(&time);
    matmul(input_A, input_B, output, m, n, k);
    stop(&time);
    printf("Elapsed time (ms): ");
    print(&time);
    printf("\n");

    printf("Correctness: ");

    //compare the result with the reference matrix
    int flag = 0;
    for(int x=0; x<m; x++){
        for(int y=0; y<n; y++){
            if(output[x * n + y] != output_ref[x * n + y]) {
                flag = 1;
            }
        }
    }

    if(flag == 1) printf("FAIL\n");
    else printf("PASS\n");

    free(input_A);
    free(input_B);
    free(output);
    free(output_ref);
}