#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --cpus-per-task=12       # CPU cores/threads
#SBATCH --mem=50000M               # memory per node
#SBATCH --time=0-23:05
#SBATCH --output=./slurm_f/xxie92-%A_%a.out
#SBATCH --array=0-2 # job array index




files=( '6nmv_H_1_113_VH.pdb' '6hga_H_1_113_VH.pdb' '1i9r_K_1_113_VH.pdb')


python convert_6d_seq_to_pdb.py singlecdr_inpaint_h3_Jun_2024_fixed_padding/samples_${SLURM_ARRAY_TASK_ID}.pkl single_hv_example_pdb/${files[${SLURM_ARRAY_TASK_ID}]}  ${SLURM_ARRAY_TASK_ID}  3

#python convert_6d_seq_to_pdb.py singlecdr_inpaint_h1_Jun_2024_fixed_padding/samples_${SLURM_ARRAY_TASK_ID}.pkl single_hv_example_pdb/${files[${SLURM_ARRAY_TASK_ID}]}  ${SLURM_ARRAY_TASK_ID}  1

#python convert_6d_seq_to_pdb.py singlecdr_inpaint_h2_Jun_2024_fixed_padding/samples_${SLURM_ARRAY_TASK_ID}.pkl single_hv_example_pdb/${files[${SLURM_ARRAY_TASK_ID}]}  ${SLURM_ARRAY_TASK_ID}  2





