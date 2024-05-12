#!/bin/bash
#SBATCH --partition=mrigpu
#SBATCH --job-name=run_unet
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1000:00:00

#SBATCH --output=outputs/%j.out

# Check if the model argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

# Assign the value of the model argument to a variable
model=$1

conda activate ls

# Uncomment this line if you want to create the output directory
# mkdir -p "$full_path"

nvidia-smi
# watch -n 1 nvidia-smi
module use /mnt/it_software/easybuild/modules/all
ml avail

# Run the Python script based on the specified model
case "$model" in
    "unet")
        python scripts/unet.py
        ;;
    "attn_unet")
        python scripts/att_unet_2d_code.py
        ;;
    "swin_unet")
        python scripts/swinunet.py
        ;;
    "trans_unet")
        python scripts/transunet.py
        ;;
    "deeplabv3")
        python scripts/deeplabv3.py
    ;;
    # Add more cases for other models as needed
    *)
        echo "Unsupported model: $model"
        exit 1
        ;;
esac

