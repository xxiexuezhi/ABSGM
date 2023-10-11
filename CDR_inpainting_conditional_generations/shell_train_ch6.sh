#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --gres=gpu:p100:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=12       # CPU cores/threads
#SBATCH --mem=50000M               # memory per node
#SBATCH --time=0-23:05
module load cuda
python train_cdr_for_seq_generation.py configs/inpainting_ch6.yml --resume protein_inverse 
#python train_cdr_ch6_with_no_epitope2.py configs/inpainting_ch6.yml 

#python train_transfer_learning_cdr_ch6_with_epitope2.py configs/inpainting_ch6_lr05.yml


