# Block matrix multiplication using MPI

## Two approaches are used
- Point-to-point communications (`scatter` and `gather`)
- Collective communications

## Build
```
mpicc -o collective -Wall collective.c common.c
mpicc -o p2p -Wall p2p.c common.c
```

## Run
mpirun -hostfile hostfile -np 4 ./collective
mpirun -hostfile hostfile -np 4 ./p2p

## Note
The two programs can only run with  4 processes, no more, no less.
