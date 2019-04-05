#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 10;                       /* Square matrix size */
double *a, *b, *c;                      /* Data blocks init in root */
double *a_block, *b_block, *c_block;    /* Blocks to calculate on each process */

double* alloc_matrix(int size) {
    return (double*) malloc(size * size * sizeof(double));
}


void init_matrix(double* matrix, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i*size + j] = j + 1;
        }
    }
}

void fill_matrix(double* matrix, int size, int fill_value) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i*size + j] = fill_value;
        }
    }
}


void print_matrix(double* matrix, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            printf("% 6.0lf ", matrix[i*size + j]);
        printf("\n");
    }
    printf("-------------------------------------------------\n");
}


void block_multiply(int block_size) {
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

    if (world_rank == 0) {
        a = alloc_matrix(N);
        b = alloc_matrix(N);
        init_matrix(a, N);
        init_matrix(b, N);
    }

    a_block = alloc_matrix(N / 2);
    b_block = alloc_matrix(N / 2);
    c_block = alloc_matrix(N / 2);

    MPI_Datatype array_block;
    if (world_rank == 0) {
        int sizes[2] = {N, N};
        int subsizes[2] = {N/2, N/2};
        int starts[2] = {0, 0};
        MPI_Type_create_subarray(2, sizes, subsizes, starts,
            MPI_ORDER_C, MPI_DOUBLE, &array_block);

        int double_size;
        MPI_Type_size(MPI_DOUBLE, &double_size);
        MPI_Type_create_resized(array_block, 0, 1*double_size, &array_block);
        MPI_Type_commit(&array_block);

        c = alloc_matrix(N);
        fill_matrix(c, N, 0);

    }
    
    /********************************************************/
    /* Do multiplications on submatrices */
    /*
            A               B
        | 0 | 1 |       | 0 | 1 |
        ---------       ---------
        | 2 | 3 |       | 2 | 3 |

    */
    const int blocks[4] = {
        0, N/2,
        N*(N/2), N*(N/2) + N/2
    };

    int send_counts[4] = {1, 1, 1, 1};


    /*
        First pass:
        Proc 0: a0 * b0
        Proc 1: a0 * b1
        Proc 2: a2 * b0
        Proc 3: a3 * b1
    */
    {
        /* Send one block of a to each process */
        int a_blocks_indices[4] = {blocks[0], blocks[0], blocks[2], blocks[2]};
        MPI_Scatterv(a, send_counts, a_blocks_indices, array_block,
            a_block, (N/2)*(N/2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Send one block of b to each process */
        int b_block_indices[4] = {blocks[0], blocks[1], blocks[0], blocks[1]};
        MPI_Scatterv(b, send_counts, b_block_indices, array_block,
            b_block, (N/2)*(N/2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    /* Multiply the first pair of blocks */
    block_multiply(N / 2);

    /*
        Second pass:
        Proc 0: a1 * b2
        Proc 1: a1 * b3
        Proc 2: a3 * b2
        Proc 3: a3 * b3
    */
    /* Send remaining block of a */
    {
        int a_blocks_indices[4] = {blocks[1], blocks[1], blocks[3], blocks[3]};
        MPI_Scatterv(a, send_counts, a_blocks_indices, array_block,
            a_block, (N/2)*(N/2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int b_block_indices[4] = {blocks[2], blocks[3], blocks[2], blocks[3]};
        MPI_Scatterv(b, send_counts, b_block_indices, array_block,
            b_block, (N/2)*(N/2), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    /* Multiply the remaining pair of blocks */
    block_multiply(N / 2);


    /********************************************************/
    /* Send the results back to process 0 */
    int recv_counts[4] = {1, 1, 1, 1};
    MPI_Gatherv(c_block, (N/2)*(N/2), MPI_DOUBLE, 
        c, recv_counts, blocks, array_block, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        print_matrix(a, N);
        print_matrix(c, N);
    }

    MPI_Finalize();
}