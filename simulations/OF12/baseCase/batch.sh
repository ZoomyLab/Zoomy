#!/usr/bin/zsh

### Job name
#SBATCH --job-name=FOAM_SERIAL

### Request the time you need for execution. The full format is D-HH:MM:SS
### You must at least specify minutes or days and hours and may add or
### leave out any other parameters
#SBATCH --time=0-01:00:00

### Request 12 processes, all on a single node
#SBATCH --nodes=1
#SBATCH --ntasks=8

### Load the required module files
module load GCC/13.3.0
module load OpenMPI/5.0.3
module load OpenFOAM/v2406

### start the OpenFOAM binary in parallel, cf.
$MPIEXEC $FLAGS_MPI_BATCH interFoam -parallel
