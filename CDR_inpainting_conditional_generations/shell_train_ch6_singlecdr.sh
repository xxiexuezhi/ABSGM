#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --gres=gpu:v100:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=12       # CPU cores/threads
#SBATCH --mem=60000M               # memory per node
#SBATCH --time=0-23:05
#SBATCH --output=./slurm_f/xxie92-%A_%a.out
#SBATCH --array=1-3 # job array index

module load cuda

#python single_cdr_train_cdr_ch6_with_epitope2.py  configs/inpainting_ch6.yml --pkl mask_only_H1_protein_dataset_dataset3_l_with_no_epitope_match_cdr_10k_all_matching_info_update_into_all_data_for_top_left_positions_with_no_padding_Mar10th.pkl
#python single_cdr_train_cdr_ch6_with_epitope2.py  configs/inpainting_ch6.yml --pkl mask_only_H2_protein_dataset_dataset3_l_with_no_epitope_match_cdr_10k_all_matching_info_update_into_all_data_for_top_left_positions_with_no_padding_Mar10th.pkl
# please note, the previous two are no epitope info. the bleow one is with epitotpe info.
python single_cdr_train_cdr_ch6_with_epitope2.py  configs/inpainting_ch6.yml --pkl H_chain_only_mask_only_H${SLURM_ARRAY_TASK_ID}_protein_dataset_dataset3_l_with_epitope_match_cdr_4k_all_matching_info_add_after_masking_and_epitope_info_no_padding_midptr_64_Mar10_2023.pkl


