#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --ntasks-per-node=32
# #SBATCH --ntasks-per-node=44
#SBATCH --mem=4G
# #SBATCH --mem=110G
# #SBATCH --mem=180G
#SBATCH --account=rrg-pmkim
#SBATCH --job-name=protein-sgm
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --output=logs/protein-sgm-%N-%j.log
#SBATCH --error=logs/protein-sgm-%N-%j.err

set -ev

unset XDG_RUNTIME_DIR

module load singularity
# For DSSP
module load StdEnv/2020 gcc/9.3.0 dssp/3.1.4
host=$(hostname)
if [[ $host =~ "blg" ]] ; then
  SINGULARITY_BINDS="--bind /lustre01 --bind /lustre02 --bind /lustre03 --bind /lustre04"
elif [[ $host =~ "nia" ]] ; then
  SINGULARITY_BINDS="--bind /scratch --bind /project --bind /gpfs"
elif [[ $host =~ "ng" ]] || [[ $host =~ "nc" ]] ; then
  SINGULARITY_BINDS="--bind /scratch --bind /project --bind /lustre05 --bind /lustre06 --bind /lustre07 --bind /localscratch"
else
  SINGULARITY_BINDS="--bind /scratch --bind /project"
fi
echo $host $SINGULARITY_BINDS

singularity exec --bind /cvmfs ${SINGULARITY_BINDS} \
  ~/singularity/protein-sgm.sif \
  bash -c "
source /opt/conda/etc/profile.d/conda.sh;
conda activate protein-sgm;
jupyter lab --ip 0.0.0.0 --port 9000 --no-browser;"

