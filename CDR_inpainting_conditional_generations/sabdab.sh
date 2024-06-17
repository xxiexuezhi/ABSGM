#!/bin/bash
#SBATCH --account=def-pmkim
#SBATCH --cpus-per-task=12       # CPU cores/threads
#SBATCH --mem=50000M               # memory per node
#SBATCH --time=0-43:05
python2 sabdab_downloader.py -s 20230213_0908605_summary.tsv -o pdb/ --chothia_pdb --annotation   --sequences --imgt

