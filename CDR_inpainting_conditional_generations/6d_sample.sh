#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --gres=gpu:v100:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=8       # CPU cores/threads
#SBATCH --mem=50000M               # memory per node
#SBATCH --time=0-23:05
#SBATCH --output=./slurm_f/xxie92-%A_%a.out
#SBATCH --array=0-11 # job array index


module load cuda

for i in {1,2,3}
do
	python sampling_6d.py ./configs/inpainting_ch6.yml ../saved_weights/h${i}_inpaint.pth --pkl proteindataset_benchmark_half_12testset_h${i}.pkl --chain A --index ${SLURM_ARRAY_TASK_ID}  --tag singlecdr_inpaint_h${i}_Jun_2024_fixed_padding
done  
