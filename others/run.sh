#!/bin/bash
#SBATCH --partition=mrigpu
#SBATCH --job-name=basic_slurm_job
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

#SBATCH --output=/home/sureshp/Landslide/Suresh/outputs/%j.out


# GENERATE FOLDER NAME
output_folder_path="/home/sureshp/Landslide/Suresh/outputs/"
folder_name="$(date +'%Y%m%d')_${SLURM_JOB_ID}"
full_path="$output_folder_path$folder_name"

mkdir -p "$full_path"

nvidia-smi
module use /mnt/it_software/easybuild/modules/all
ml avail

python mycode.py "$full_path"

mv outputs/*.out outputs/"$folder_name"
