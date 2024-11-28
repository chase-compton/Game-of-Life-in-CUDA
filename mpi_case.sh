#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load openmpi/4.1.4-gcc11

mpicc -O3 -o mpi_gol mpi_game_of_life.c -lm


mpirun -np 20 ./mpi_gol 10000 5000 20 /home/ualclsd0146/HW5_CS481/output
mpirun -np 20 ./mpi_gol 10000 5000 20 /home/ualclsd0146/HW5_CS481/output
mpirun -np 20 ./mpi_gol 10000 5000 20 /home/ualclsd0146/HW5_CS481/output









