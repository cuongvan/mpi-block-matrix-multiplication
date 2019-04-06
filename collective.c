#include <mpi.h>
#include "common.h"
#include <stdio.h>
#include <stdlib.h>

const int N = 10;                     /* Square matrix size */
double *a, *b, *c;                      /* Data blocks init in root */
double *a_block, *b_block, *c_block;    /* Blocks to calculate on each process */

int main(int argc, char** argv) {
    int world_size, world_rank;
    const int block_size = N/2;
    const int num_block_elements = block_size * block_size;

    MPI_Init(&argc, &argv);
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
    const int blocks[4] = {
        0, block_size,
        N*block_size, N*block_size + block_size
    };

    int send_counts[4] = {1, 1, 1, 1};


    /* First pass:
        Proc 0: a0 * b0
        Proc 1: a0 * b1
        Proc 2: a2 * b0
        Proc 3: a3 * b1 */
    {
        /* Send one block of a to each process */
        int a_blocks_indices[4] = {blocks[0], blocks[0], blocks[2], blocks[2]};
        MPI_Scatterv(a, send_counts, a_blocks_indices, array_block,
            a_block, num_block_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* Send one block of b to each process */
        int b_block_indices[4] = {blocks[0], blocks[1], blocks[0], blocks[1]};
        MPI_Scatterv(b, send_counts, b_block_indices, array_block,
            b_block, num_block_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    /* Multiply the first pair of blocks */
    block_multiply(a_block, b_block, c_block, block_size);

    /* Second pass:
        Proc 0: a1 * b2
        Proc 1: a1 * b3
        Proc 2: a3 * b2
        Proc 3: a3 * b3 */
    {
        int a_blocks_indices[4] = {blocks[1], blocks[1], blocks[3], blocks[3]};
        MPI_Scatterv(a, send_counts, a_blocks_indices, array_block,
            a_block, num_block_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int b_block_indices[4] = {blocks[2], blocks[3], blocks[2], blocks[3]};
        MPI_Scatterv(b, send_counts, b_block_indices, array_block,
            b_block, num_block_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    /* Multiply the remaining pair of blocks */
    block_multiply(a_block, b_block, c_block, block_size);

    /********************************************************/
    /* Send the results back to process 0 */
    MPI_Gatherv(c_block, num_block_elements, MPI_DOUBLE, 
        c, send_counts, blocks, array_block, 0, MPI_COMM_WORLD);

    free(a_block);
    free(b_block);
    free(c_block);

    /* End of paralle section */
    MPI_Barrier(MPI_COMM_WORLD);
    time += MPI_Wtime();

    if (world_rank == 0) {
        if (N <= 10) {
            printf("Input matrices:\n");
            print_matrix(a, N);
            printf("Output matrix:\n");
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