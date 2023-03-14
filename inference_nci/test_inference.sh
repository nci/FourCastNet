set -eu
curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON=$curr_dir/../python_env/bin/python

pretrained_root=/g/data/wb00/admin/staging/FourCastNet/v0
output_path=$curr_dir/test_outputs/preds.zarr

$PYTHON -u $curr_dir/inference.py \
  --start_time=2018-01-01T00:00:00 \
  --end_time=2018-01-02T00:00:00 \
  --checkpoint_dir=$pretrained_root/pretrained/model_weights/FCN_weights_v0 \
  --stats_dir=$pretrained_root/data/additional/stats_v0 \
  --output_path=$output_path \
  --prediction_length=3
