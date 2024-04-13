#!/bin/bash
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -J openmp_3d
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=64
export OMP_PLACES=cores

#run the application:
srun ./openmp_3d > out_openmo3d.out


