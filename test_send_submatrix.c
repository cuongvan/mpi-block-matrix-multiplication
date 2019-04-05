#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 10;
double **a, **tmp;

double** matrix_alloc(int nrows, int ncols) {
    double** matrix;
    int i;
    matrix = (double**) malloc(nrows * sizeof(double *));
    for (i = 0; i < nrows; i++)
        matrix[i] = (double*) malloc(ncols * sizeof(double));
    return matrix;
}

void init_matrix(double** matrix, int nrows, int ncols) {
    int i, j;
    for (i = 0; i < nrows; i++) {
        for (int j = 0; j < nrows; j++)
            matrix[i][j] = i + j;
    }
}

void print_matrix(double** matrix, int nrows, int ncols) {
    int i, j;
    for (i = 0; i < nrows; i++) {
        for (int j = 0; j < nrows; j++)
            printf("% 6.0lf ", matrix[i][j]);
        printf("\n");
    }
}

MPI_Datatype create_block_type(int row, int col, int sub_size) {
    MPI_Datatype new_type;

    int starts = {row, col};
    int big_sizes = {N, N};
    int sub_sizes = {sub_size, sub_size};
    MPI_Type_create_subarray(2, big_sizes, sub_size, starts, MPI_ORDER_C, MPI_INT, &new_type);
    MPI_Type_commit(new_type);

    return new_type;
}

MPI_Datatype create_vector_type(int nrows, int ncols) {
}

int main() {
    int world_size, world_rank;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        a = matrix_alloc(N, N);
        init_matrix(a, N, N);
        // print_matrix(a, N, N);
        MPI_Datatype top_left_block = create_block_type(0, 0, N / 2);
        MPI_Recv(&a[5][5], 1, top_left_block, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    else {
        tmp = matrix_alloc(N/2, N/2);
        // MPI_Send(tmp, )
    }



    MPI_Finalize();
}