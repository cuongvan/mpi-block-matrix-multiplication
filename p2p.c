#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 10;                     /* Square matrix size */
double *a, *b, *c;                      /* Data blocks init in root */
double *a_block, *b_block, *c_block;    /* Blocks to calculate on each process */

double* alloc_matrix(int size) {
    return (double*) malloc(size * size * sizeof(double));
}

void init_matrix(double* matrix, int size) {
    int i, j;
    for (i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i*size + j] = 2*(i/(size/2)) + j/(size/2);
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
    printf("-------------------------------------------------------------------------\n");
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
    const int block_size = N / 2;


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
        c = alloc_matrix(N);
        init_matrix(a, N);
        init_matrix(b, N);
        fill_matrix(c, N, 0);
    }

    /* Start of parallel section */
    MPI_Barrier(MPI_COMM_WORLD);
    double time = - MPI_Wtime();

    a_block = alloc_matrix(block_size);
    b_block = alloc_matrix(block_size);
    c_block = alloc_matrix(block_size);
    fill_matrix(a_block, block_size, 0);
    fill_matrix(b_block, block_size, 0);
    fill_matrix(c_block, block_size, 0);

    MPI_Datatype array_block;
    if (world_rank == 0) {
        int sizes[2] = {N, N};
        int subsizes[2] = {block_size, block_size};
        int starts[2] = {0, 0};
        MPI_Type_create_subarray(2, sizes, subsizes, starts,
            MPI_ORDER_C, MPI_DOUBLE, &array_block);

        int double_size;
        MPI_Type_size(MPI_DOUBLE, &double_size);
        MPI_Type_create_resized(array_block, 0, 1*double_size, &array_block);
        MPI_Type_commit(&array_block);
    }
    
    /********************************************************/
    /* Do multiplications on submatrices */
    /*
            A               B
        | 0 | 1 |       | 0 | 1 |
        ---------       ---------
        | 2 | 3 |       | 2 | 3 |

    */
    const int blocks[4] = {         /* Memory address of top-left corners */
        0, N/2,                     /*   of blocks 0, 1, 2, 3 */
        N*(N/2), N*(N/2) + N/2
    };

    /* First pass:
        Proc 0: a0 * b0
        Proc 1: a0 * b1
        Proc 2: a2 * b0
        Proc 3: a3 * b1 */
    {
        const int a_tag = 0, b_tag = 1;
        if (world_rank == 0) {
            MPI_Request requests[6];
            
            MPI_Isend(&a[blocks[0]], 1, array_block, 1, a_tag, MPI_COMM_WORLD, &requests[0]);     /* a0 --> proc 1 */
            MPI_Isend(&a[blocks[2]], 1, array_block, 2, a_tag, MPI_COMM_WORLD, &requests[1]);     /* a2 --> proc 2 */
            MPI_Isend(&a[blocks[2]], 1, array_block, 3, a_tag, MPI_COMM_WORLD, &requests[2]);     /* a2 --> proc 3 */

            MPI_Isend(&b[blocks[1]], 1, array_block, 1, b_tag, MPI_COMM_WORLD, &requests[3]);     /* b1 --> proc 1 */
            MPI_Isend(&b[blocks[0]], 1, array_block, 2, b_tag, MPI_COMM_WORLD, &requests[4]);     /* b0 --> proc 2 */
            MPI_Isend(&b[blocks[1]], 1, array_block, 3, b_tag, MPI_COMM_WORLD, &requests[5]);     /* b1 --> proc 3 */


            /* proc 0: copy a0, b0 --> a_block, b_block */
            int i, j;
            for (i = 0; i < block_size; i++) {
                for (j = 0; j < block_size; j++) {
                    a_block[i*block_size + j] = a[blocks[0] + i*N + j];
                    b_block[i*block_size + j] = b[blocks[0] + i*N + j];
                }
            }

            MPI_Waitall(6, requests, MPI_STATUSES_IGNORE);
        }
        else {
            MPI_Request requests[2];
            MPI_Irecv(a_block, block_size * block_size, MPI_DOUBLE, 0, a_tag, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(b_block, block_size * block_size, MPI_DOUBLE, 0, b_tag, MPI_COMM_WORLD, &requests[1]);

            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }
    }

    block_multiply(block_size);
    MPI_Barrier(MPI_COMM_WORLD);    // Every slave procs have to be done before `a_block` can be reused

    /*  Second pass:
            Proc 0: a1 * b2
            Proc 1: a1 * b3
            Proc 2: a3 * b2
            Proc 3: a3 * b3 */
    {
        const int a_tag = 0, b_tag = 1;
        if (world_rank == 0) {
            MPI_Request requests[6];
            
            MPI_Isend(&a[blocks[1]], 1, array_block, 1, a_tag, MPI_COMM_WORLD, &requests[0]);     /* a0 --> proc 1 */
            MPI_Isend(&a[blocks[3]], 1, array_block, 2, a_tag, MPI_COMM_WORLD, &requests[1]);     /* a2 --> proc 2 */
            MPI_Isend(&a[blocks[3]], 1, array_block, 3, a_tag, MPI_COMM_WORLD, &requests[2]);     /* a2 --> proc 3 */

            MPI_Isend(&b[blocks[3]], 1, array_block, 1, b_tag, MPI_COMM_WORLD, &requests[3]);     /* b1 --> proc 1 */
            MPI_Isend(&b[blocks[2]], 1, array_block, 2, b_tag, MPI_COMM_WORLD, &requests[4]);     /* b0 --> proc 2 */
            MPI_Isend(&b[blocks[3]], 1, array_block, 3, b_tag, MPI_COMM_WORLD, &requests[5]);     /* b1 --> proc 3 */


            /* proc 0: copy a1, b2 --> a_block, b_block */
            int i, j;
            for (i = 0; i < block_size; i++) {
                for (j = 0; j < block_size; j++) {
                    a_block[i*block_size + j] = a[blocks[1] + i*N + j];
                    b_block[i*block_size + j] = b[blocks[2] + i*N + j];
                }
            }

            MPI_Waitall(6, requests, MPI_STATUSES_IGNORE);
        }
        else {
            MPI_Request requests[2];
            MPI_Irecv(a_block, block_size * block_size, MPI_DOUBLE, 0, a_tag, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(b_block, block_size * block_size, MPI_DOUBLE, 0, b_tag, MPI_COMM_WORLD, &requests[1]);

            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }
    }

    block_multiply(block_size);

    /* Send all result to proc 0 */
    {
        const int tag = 0;
        if (world_rank == 0) {
            MPI_Request requests[3];
            MPI_Irecv(&c[blocks[1]], 1, array_block, 1, tag, MPI_COMM_WORLD, &requests[0]);
            MPI_Irecv(&c[blocks[2]], 1, array_block, 2, tag, MPI_COMM_WORLD, &requests[1]);
            MPI_Irecv(&c[blocks[3]], 1, array_block, 3, tag, MPI_COMM_WORLD, &requests[3]);

            /* proc 0: copy c_block --> c0 */
            int i, j;
            for (i = 0; i < block_size; i++)
                for (j = 0; j < block_size; j++)
                    c[blocks[0] + i*N + j] = c_block[i* block_size + j];

            MPI_Waitall(3, requests, MPI_STATUSES_IGNORE);
        }
        else {
            MPI_Send(c_block, block_size * block_size, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
        }
    }


    free(a_block);
    free(b_block);
    free(c_block);

    /* End of paralle section */
    MPI_Barrier(MPI_COMM_WORLD);
    time += MPI_Wtime();

    if (world_rank == 0) {
        if (N <= 10) {
            print_matrix(a, N);
            print_matrix(c, N);
        }
        printf("Elapsed time: %.2lf seconds\n", time);
    }

    if (world_rank == 0) {
        free(a);
        free(b);
        free(c);
        MPI_Type_free(&array_block);
    }

    MPI_Finalize();
    return 0;
}