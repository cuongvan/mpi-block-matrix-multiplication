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

## Results in seconds (avarage of 5 runs)
### On Dell, 1.8 GHz, 4 cores
| Matrix size  | P2P   | Collective |
|-------------:|------:|-----------:|
|          100 | 0.006 |      0.004 |
|          200 |  0.02 |       0.02 |
|          400 |  0.15 |      0.146 |
|          800 | 1.838 |      2.204 |
|         1000 | 5.856 |      5.692 |

### On Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz
| Matrix size  | P2P   | Collective |
|-------------:|------:|-----------:|
|          100 |  0.00 |       0.00 |
|          200 |  0.01 |       0.01 |
|          400 |  0.06 |       0.06 |
|          800 |  0.48 |       0.47 |
|         1000 |  0.96 |       0.95 |
|         2000 |  7.06 |      7.036 |