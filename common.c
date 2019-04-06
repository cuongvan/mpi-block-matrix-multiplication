#include "common.h"
#include <stdio.h>
#include <stdlib.h>

double* alloc_matrix(int size) {
    return (double*) malloc(size * size * sizeof(double));
}

void init_matrix(double* matrix, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            matrix[i*size + j] = 2*(i/(size/2)) + j/(size/2);
        }
    }
}

void fill_matrix(double* matrix, int size, int fill_value) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            matrix[i*size + j] = fill_value;
        }
    }
}


void print_matrix(double* matrix, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++)
            printf("% 6.0lf ", matrix[i*size + j]);
        printf("\n");
    }
    printf("-------------------------------------------------------------------------\n");
}

void block_multiply(double* a_block, double* b_block,
                    double* c_block, int block_size) {
    int i, j, k;
    double *aptr, *bptr, *cptr;

    for (i = 0; i < block_size; i++)
        for (j = 0; j < block_size; j++) {
            cptr = c_block + i*block_size + j;
            aptr = a_block + i*block_size;
            bptr = b_block + j;
            for (k = 0; k < block_size; k++) {
                *cptr += *(aptr++) * *bptr;
                bptr += block_size;
            }
        }
}