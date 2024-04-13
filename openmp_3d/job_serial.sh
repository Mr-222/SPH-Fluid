#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J serial_3d
#SBATCH -t 00:30:00

srun .serial_3d