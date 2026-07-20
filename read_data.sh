#!/bin/bash
#SBATCH --job-name=esmknot
#SBATCH --account=hggroup
#SBATCH --partition=cpu
#SBATCH -N 1 -n 8


source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm3


python /home/pdengad/esmknot/esmknot/data/process_preference_data.py