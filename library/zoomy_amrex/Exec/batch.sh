#!/usr/local_rwth/bin/zsh

#### MPI ranks
#SBATCH --ntasks=8
### threads per task=MPI rank
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH --time=0-00:30:00
#SBATCH --job-name=dam_break
#SBATCH --output=output.%J.txt

module purge
module load foss

$MPIEXEC $FLAGS_MPI_BATCH ./main3d.gnu.MPI.ex inputs
