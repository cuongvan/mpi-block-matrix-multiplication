#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 600;
double *a, *b, *c;

void alloc_matrices() {
    a = (double*) malloc(N * N * sizeof(double));
    b = (double*) malloc(N * N * sizeof(double));
    c = (double*) malloc(N * N * sizeof(double));
}

void init_matrices() {
    int i, j;
    for (i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i*N + j] = j + 1;
            b[i*N + j] = j + 1;
            c[i*N + j] = 0;
        }
            
    }
}

void free_matrices() {
    free(a);
    free(b);
    free(c);
}

void print_matrix(double* matrix) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("% 6.0lf ", matrix[i*N + j]);
        printf("\n");
    }
    printf("-------------------------------------------------\n");
}

MPI_Datatype create_block_type(int row, int col, int sub_size) {
    MPI_Datatype new_type;

    int starts[2] = {row, col};
    int big_sizes[2] = {N, N};
    int sub_sizes[2] = {sub_size, sub_size};
    MPI_Type_create_subarray(2, big_sizes, sub_sizes, starts, MPI_ORDER_C, MPI_INT, &new_type);
    MPI_Type_commit(&new_type);

    return new_type;
}

void mm(int crow, int ccol, /* Corner of C block */
        int arow, int acol, /* Corner of A block */
        int brow, int bcol, /* Corner of B block */
        int block_size)     /* Block size */
{
    int i, j, k;
    double *aptr, *bptr, *cptr;

    for (i = 0; i < block_size; i++)
        for (j = 0; j < block_size; j++) {
            cptr = c + (crow + i)*N + ccol + j;
            aptr = a + (arow + i)*N + acol;
            bptr = b + brow * N + bcol + j;
            for (k = 0; k < block_size; k++) {
                *cptr += *(aptr++) * *bptr;
                bptr += N;
            }
        }
}

void matrix_mul();

int main() {
    int world_size, world_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_size != 4) {
        if (world_rank == 0) {
            printf("Need 4 processes");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Finalize();
        exit(1);
    }

    alloc_matrices();
    init_matrices();

    MPI_Barrier(MPI_COMM_WORLD);
    double time = - MPI_Wtime();

    int crow = (N/2) * (world_rank / 2);
    int ccol = (N/2) * (world_rank % 2);
    mm(crow, ccol, crow, 0, 0, ccol, N/2);
    mm(crow, ccol, crow, N/2, N/2, ccol, N/2);

    // MPI_Reduce(c, c, N*N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double* tmp;
    if (world_rank == 0)
        tmp = (double*) malloc(N * N * sizeof(double));
    MPI_Reduce(c, tmp, N*N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    
    if (world_rank == 0) {
        free(c);
        c = tmp;
        tmp = NULL;
    }

    time += MPI_Wtime();

    if (world_rank == 0) {
        // print_matrix(c);
        printf("Elapsed time: %.2lf seconds\n", time);
    }
    
    free_matrices();
    MPI_Finalize();
}