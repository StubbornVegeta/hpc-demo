# build
```bash 
mpicxx ring_allreduce.cpp -O3 -std=c++11 -o ring_allreduce
```

# run
```bash 
mpirun -N 4 -n 4 ring_allreduce 1000 10000
```
