#!/bin/bash

#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l jobfs=200GB
#PBS -l storage=gdata/dk92+gdata/wb00+gdata/rt52+<YOUR_PROJECT_STORAGE>
#PBS -l mem=200GB
#PBS -l walltime=08:00:00
#PBS -l wd
#PBS -P <YOUR_PROJECT_HERE>

set -eu
  
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/23.03
module load cuda/11.7.0

checkpoint_dir='/g/data/wb00/FourCastNet/nvlab/v0/pretrained'
stats_dir='/g/data/wb00/FourCastNet/nvlab/v0/data/stats/'
output_path='./output'

# Run inference AFNO Backbone 
python inference/inference.py \
    --config=afno_backbone \
    --run_num=0 --vis   \
    --weights '/g/data/wb00/FourCastNet/nvlab/v0/pretrained/backbone.ckpt' \
    --override_dir './output'

# Run inference_ensemble AFNO Backbone
python inference/inference_ensemble.py \
    --config=afno_backbone \
    --run_num=0 \
    --n_pert 10 \
    --override_dir './output'  \
    --weights '/g/data/wb00/FourCastNet/nvlab/v0/pretrained/backbone.ckpt'
    
# Run inference for precipitation
python inference/inference_precip.py \
    --config=precip \
    --run_num=0 \
    --vis \
    --weights '/g/data/wb00/FourCastNet/nvlab/v0/pretrained/precip.ckpt' \
    --override_dir './output'

# Run inference to generate preds.zarr
python -u ./inference_nci/inference.py \
  --start_time=2018-1-01T18:00:00 \
  --end_time=2018-1-15T18:00:00 \
  --checkpoint_dir=$checkpoint_dir \
  --stats_dir=$stats_dir \
  --output_path=$output_path \
  --prediction_length=20
        

        

        

    

