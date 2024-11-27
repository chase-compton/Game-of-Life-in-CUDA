#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load openmpi/4.1.4-gcc11

mpicc -O3 -o mpi_gol mpi_game_of_life.c -lm

mpirun -np 1 ./mpi_gol 80 100 1 /home/ualclsd0146/HW5_CS481/output
mpirun -np 4 ./mpi_gol 80 100 4 /home/ualclsd0146/HW5_CS481/output






