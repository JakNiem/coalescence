#!/bin/sh

#SBATCH -J coa_d3_T7.0
#SBATCH --nodes=1

### 1*8 MPI ranks
#SBATCH --ntasks=1

### 128/16 MPI ranks per node
#SBATCH --ntasks-per-node=1

### tasks per MPI rank
#SBATCH --cpus-per-task=1

#SBATCH -e ./err_drop.%j.log
#SBATCH -o ./out_drop.%j.log

/home/niemann/ls1-mardyn_cylindricSampling/build/src/MarDyn --final-checkpoint=1 config_drop.xml 
    