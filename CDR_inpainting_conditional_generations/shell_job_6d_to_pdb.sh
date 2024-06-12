#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --cpus-per-task=12       # CPU cores/threads
#SBATCH --mem=50000M               # memory per node
#SBATCH --time=0-23:05
#SBATCH --output=./slurm_f/xxie92-%A_%a.out
#SBATCH --array=0-11 # job array index




files=( '3hmx_H_1_113_VH.pdb' '7zfb_C_1_113_VH.pdb' '7mep_H_1_113_VH.pdb' '6nz7_G_1_113_VH.pdb' '6nn3_H_1_111_VH.pdb' '6nmv_H_1_113_VH.pdb' '6hga_H_1_113_VH.pdb' '1i9r_K_1_113_VH.pdb' '5kw9_H_1_113_VH.pdb' '4ydl_B_1_113_VH.pdb' '6bp2_H_1_113_VH.pdb' '7eng_H_1_113_VH.pdb' )



python convert_6d_seq_to_pdb.py ../proteinsgm_singlecdr_inpaint_retrain/singlecdr_inpaint_h1_Jun_2024/samples_${SLURM_ARRAY_TASK_ID}.pkl half_h3_single_heavy_benckmark_pdb/${files[${SLURM_ARRAY_TASK_ID}]}  ${SLURM_ARRAY_TASK_ID}  1

python convert_6d_seq_to_pdb.py ../proteinsgm_singlecdr_inpaint_retrain/singlecdr_inpaint_h2_Jun_2024/samples_${SLURM_ARRAY_TASK_ID}.pkl half_h3_single_heavy_benckmark_pdb/${files[${SLURM_ARRAY_TASK_ID}]}  ${SLURM_ARRAY_TASK_ID}  2

python convert_6d_seq_to_pdb.py ../proteinsgm_singlecdr_inpaint_retrain/singlecdr_inpaint_h3_Jun_2024/samples_${SLURM_ARRAY_TASK_ID}.pkl half_h3_single_heavy_benckmark_pdb/${files[${SLURM_ARRAY_TASK_ID}]}  ${SLURM_ARRAY_TASK_ID}  3





