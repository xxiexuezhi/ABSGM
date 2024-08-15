# Antibody-SGM


![Antibody-SGM schematic](/banner_toc.png)






## Descriptions


Antibody-SGM is a score-based generative modeling for de novo antibody heavy chain design. This repository contains the codebase for  [Antibody-SGM, A score-based generative model for de novo antibody heavy chain design]




## Installation


Run the following commands to install AB-SGM and the necessary dependencies; installation should take less than 15 minutes. We recommend using the conda environment supplied in this repository.

1. Clone repository ` git clone https://github.com/xxiexuezhi/ABSGM.git ' or download the zip files and extract all.
2. Install conda environment `create -f ab_env.yaml`
3. Activate conda environment `conda activate ab_env`
   
To run AB-SGM, download and extract the model parameters,[Saved weights](https://drive.google.com/drive/folders/1w1yPn3rYz04p9eejsr15bJN6K7kVzSAg?usp=sharing)


The code is tested on Python 3.8.17 and needs the Pytorch GPU version for training and sampling. We are using a GPU for model training, 6D coordinate sampling, and CPU batch processing for Rosetta. More specifically, we used a single NVIDIA V100/A100 for training and inference and at least 2 core CPU/8GB RAM per Rosetta job. Please refer to the shell script inside GitHub for detailed configurations.




## Inference (Conditional generation)


All related codes are in the CDR_inpainting_conditional_generations directory. Structures are generated by sampling 6D coordinates from the model and running PyRosetta.
The encoded examples (pdb id: 6nmv, 6hga, 1i9r) are uploaded. They can be found from:


The encoded examples (pdb id: 6nmv, 6hga, 1i9r) are uploaded. you could find from 


* H1 inpaint encoded examples. [this link](https://drive.google.com/file/d/1CrvTYp0YYwrstxpf-yIi3dwyE3Ff-3pd/view?usp=sharing)
* H2 inpaint encoded examples.[this link](https://drive.google.com/file/d/1yI8h2cghVjszUH3E5YU8mEblCVtqGR9w/view?usp=sharing)
* H3 inpaint encoded examples. [this link](https://drive.google.com/file/d/120QGm0jyhFzulqILnAWIB9dA6zthHAbn/view?usp=sharing)


To encode the antigen-antibody complex structures, please refer to the readme inside the CDR_inpainting_conditional_generations/encoding/.





### 6d and sequence pairs generations


Please use python sampling_6d.py to generate 6d coordinates and sequences. We used the shell scripts for generations (please refer to h3_6d_sample.sh for more details). For instance,
```
python sampling_6d.py ./configs/inpainting_ch6.yml ../saved_weights/h3_inpaint.pth --pkl proteindataset_example_singlecdr_inpaint_h3_6nmv_6hga_1i9r.pkl --chain A --index 1  --tag singlecdr_inpaint_h3_Jun_2024_fixed_padding


```
The descriptions of each parameter are as below:


  * ./configs/inpainting_ch6.yml are the config files containing hyperparameters like batch size and data dimensions.
  
  * ../saved_weights/h3_inpaint.pth This is to load the saved weight.
  
  * --pkl proteindataset_example_singlecdr_inpaint_h3_6nmv_6hga_1i9r.pkl is the pickle file containing all encoded data with H3 regions indicated to be masked out.


  * --index 1 refers to the index number inside the proteindataset_example_singlecdr_inpaint_h3_6nmv_6hga_1i9r.pkl to be generated. This index number would match the generated file number; for example, the generated file is named samples_index.pkl.


  * --tag  singlecdr_inpaint_h3_Jun_2024_fixed_padding is the generated folder name. The code would create this folder under the current directory and store all the generated data.




6D coordinate sampling should ~ take 1 minute per sample on a normal GPU. Rosetta minimization should take about 3 hours per iteration, depending on the selected H1, H2, or H3 region size for design.






### 6d and sequence pairs to pdbs




 Please use python 'convert_6d_seq_to_pdb.py' to convert the 6d coordinates and sequence pairs into the pdb using PyRosetta. Please refer to h3_shell_job_6d_to_pdb.sh for more details. For instance, 


 
 ```


python convert_6d_seq_to_pdb.py singlecdr_inpaint_h3_Jun_2024_fixed_padding/samples_${SLURM_ARRAY_TASK_ID}.pkl single_hv_example_pdb/${files[${SLURM_ARRAY_TASK_ID}]}  ${SLURM_ARRAY_TASK_ID}  3




```




* samples_${SLURM_ARRAY_TASK_ID}.pkl is the 6d data generated with the previous shell script. 
single_hv_example_pdb/${files[${SLURM_ARRAY_TASK_ID}]} is the pdb files read in by PyRosetta to do inpainting. This file is needed to avoid doing any superposing. 
* ${SLURM_ARRAY_TASK_ID} is just the index file number for the pkl file. 
*  "3". The last number (1, 2, or 3) indicates the inpainting regions (h1,  h2, or h3).
* The PyRosetta protocol saves all iterations and intermediate structures to subdirectories in outPath. The default name is test_single_cdr_inpaint_generations_single_h[123]. The final minimized structure can be found under the outPath with the name b_n1_n2.pdb. n1 matching ${SLURM_ARRAY_TASK_ID}, the index inside the pickle files, while n2 is just the index for the generated files



## Dataset 
---
Raw antigen-antibody complex dataset for CDR conditional generations can be downloaded [here](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/)


The 'sabdab_downloader.py' is also available for download.



### Reference
---
Antibody-SGM (to be updated)
