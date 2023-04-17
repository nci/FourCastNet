#!/bin/bash

#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l jobfs=200GB
#PBS -l storage=gdata/dk92+gdata/wb00+(INSERT ANY ADDITIONAL STORAGE HERE)
#PBS -l mem=48GB
#PBS -l walltime=08:00:00
#PBS -P INSERT PROJECT

set -eu
  
module use /g/data/dk92/apps/Modules/modulefiles
module load NCI-ai-ml/22.11
module load cuda/11.7.0

pretrained_root=<pre-trained root>
output_path=<output path>

# Run inference AFNO Backbone 
python inference/inference.py \
    --config=afno_backbone \
    --run_num=0 --vis   \
    --weights '/g/data/wb00/admin/staging/FourCastNet/nvlab/v0/pretrained/backbone.ckpt' \
    --override_dir './output'

# Run inference_ensemble AFNO Backbone
python inference/inference_ensemble.py \
    --config=afno_backbone \
    --run_num=0 \
    --n_pert 10 \
    --override_dir './output'  \
    --weights '/g/data/wb00/admin/staging/FourCastNet/nvlab/v0/pretrained/backbone.ckpt'
    
# Run inference for precipitation
python inference/inference_precip.py \
    --config=precip \
    --run_num=0 \
    --vis \
    --weights '/g/data/wb00/admin/staging/FourCastNet/nvlab/v0/pretrained/precip.ckpt' \
    --override_dir './output'

# Run inference to generate preds.zarr
python -u inference_nci/inference.py \
  --start_time=2017-12-31T18:00:00 \
  --end_time=2018-12-30T18:00:00 \
  --checkpoint_dir=$pretrained_root/FCN_weights_v0 \
  --stats_dir=$pretrained_root/stats_v0 \
  --output_path=$output_path \
  --prediction_length=20
        

        

        

    

