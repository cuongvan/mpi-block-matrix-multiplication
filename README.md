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
```
mpirun -hostfile hostfile -np 4 ./collective
mpirun -hostfile hostfile -np 4 ./p2p
```

## Note
The two programs can only run with  4 processes, no more, no less.

## Results (avarage of 5 runs)
### On Dell, 1.8 GHz, 4 cores
| Matrix size  | P2P   | Collective |
|-------------:|------:|-----------:|
|          100 | 0.006 |      0.004 |
|          200 |  0.02 |       0.02 |
|          400 |  0.15 |      0.146 |
|          800 | 1.838 |      2.204 |
|         1000 | 5.856 |      5.692 |

