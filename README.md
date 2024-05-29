## Antibody-SGM: Score-based generative modeling for *de novo* antibody heavychain design

This repository contains the codebase for [Antibody-SGM: Antigen-Specific Joint Design of Antibody Sequence and
Structure using Diffusion Models](https://icml-compbio.github.io/2023/papers/WCBICML2023_paper143.pdf)


### Installation
---

We recommend using the conda environment supplied in this repository.

1. Clone repository `git clone https://gitlab.com/mjslee0921/proteinsgm.git`
2. Install conda environment `create -f ab_env.yaml`
3. Activate conda environment `conda activate ab_env`

### Training
---
Raw antigen-antibody complex dataset for CDR condtional generations can be downloaded [here] (https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/)

Once the files have been extracted, you must change `data.dataset_path` in the configuration file to the correct path containing the training data.

The training script can be found in `train.py`. For instance, to train an unconditional model (conditioned just on length), run the following command:

`python train.py ./configs/cond_length.yml`

Conditional models can be trained by replacing `./configs/cond_length.yml` with `./configs/cond_length_inpainting.yml`

### Inference
---

#### Unconditional generation

Unconditional generation of structures is achieved by first sampling 6D coordinates and sequences from the model, and running Rosetta.

`sampling_6d.py` is used to sample 6D coordinates given a model checkpoint. For instance,

`python sampling_6d.py ./configs/cond_length.yml ./checkpoints/cond_length.pth`

This will first sample random lengths. If you want to generate a specific length, you can set `--select_length True length_index 1`, which will generate a protein of length 40 (we use 1-indexing here)

To generate structures from 6D coordinates, please refer to `sampling_rosetta.py`. For instance,

`python sampling_rosetta example/sample.pkl --index 1`

This will read the first sample (1-indexing) and run <tt>MinMover, FastDesign,</tt> and <tt>FastRelax</tt> and save all intermediate structures.

You can exclude any of the Rosetta minimization steps (except <tt>MinMover</tt>) with the flags `--fastrelax False --fastdesign False`.

We are using a GPU for 6D coordinate sampling, and CPU batch processing for Rosetta since Rosetta protocols can only use a single core per job. In our case we used a single NVIDIA V100 for training and inference, and 2 core CPU/8GB RAM per Rosetta job.

#### Conditional generation

Conditional generation additionally requires the epitope-antibody complex with the designed CDR been masked out.

To generate conditional samples, you must provide the chain id and residue indices of the regions to be masked during inference. For instance, to mask residues 10-30 and 35 in chain A, you can run the following command:

`python sampling_6d.py ./configs/cond_length_inpainting.yml ./checkpoints/cond_length_inpainting.pth --pdb example/2kl8.pdb --chain A --mask_info 10:30,35`


The generated 6D coordinates can then be minimized using Rosetta in the same manner as unconditional samples.

6D coordinate sampling should ~1 minute per sample on a normal GPU, and Rosetta minimization should take a maximum of 3 hour per iteration depending on the size of the selected region for design.


The Rosetta protocol saves all iterations and intermediate structures to subdirectories in `outPath`, including structures before FastDesign and before relaxation. The default number of iterations is 3, and the final minimized structure can be found under `outPath/.../best_run/final_structure.pdb`.



### Reference
---
Antibody-SGM: Antigen-Specific Joint Design of Antibody Sequence and Structure using Diffusion Models



# Set up Env
Install conda environment conda env create -f ab_env.yaml

Activate conda environment conda activate ab_env

# Saved weights and data
Saved weights could be access here: https://drive.google.com/drive/folders/1w1yPn3rYz04p9eejsr15bJN6K7kVzSAg?usp=sharing

AB dataset could be access here: https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/
