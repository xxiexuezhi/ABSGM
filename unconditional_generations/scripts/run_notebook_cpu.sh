#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
# #SBATCH --ntasks-per-node=44
#SBATCH --mem=110G
# #SBATCH --mem=180G
#SBATCH --account=rrg-pmkim
#SBATCH --job-name=protein-sgm
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --output=logs/ab-loop-modeling-%N-%j.log
#SBATCH --error=logs/ab-loop-modeling-%N-%j.err

set -ev

unset XDG_RUNTIME_DIR

module load singularity

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

NOTEBOOK_STEM=$(basename ${NOTEBOOK_PATH%%.ipynb})
NOTEBOOK_DIR=$(dirname ${NOTEBOOK_PATH})
OUTPUT_TAG="${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_NODELIST}-${SLURM_JOB_ID}"

echo ${NOTEBOOK_PATH} ${NOTEBOOK_STEM} ${NOTEBOOK_DIR} ${OUTPUT_TAG}

singularity exec --bind /cvmfs ${SINGULARITY_BINDS} \
  ~/singularity/protein-sgm.sif \
  bash -c "
source /opt/conda/etc/profile.d/conda.sh;
conda activate base;
papermill --no-progress-bar --log-output --kernel python3 '${NOTEBOOK_PATH}' '${NOTEBOOK_DIR}/${NOTEBOOK_STEM}-${OUTPUT_TAG}.ipynb'"

