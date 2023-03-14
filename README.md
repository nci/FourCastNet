FourCastNet Inference
=====================

This repository is forked from the official [FourCastNet](https://github.com/NVlabs/FourCastNet).

In addition to the official FourCastNet, this repository contains code that performs inference
on NCI version of ERA5 from project rt52.

Setup
-----

* Ask to join NCI project rt52 on [mancini](https://my.nci.org.au/mancini).

* Run `bash setup.sh` to set up the environment. This script sets up a Python virtualenv with all the
  required dependencies.

* The inference requires pretrained weights and input normalization statistics. These files are now at
  `/g/data/wb00/admin/staging/FourCastNet/v0`.

Notebooks
---------

We also provided several Jupyter notebooks under the `notebooks` directory.
These notebooks demonstrate:

* Baseline inference on the out-of-distribution sample data provided by the original authors
  The code for baseline inference is under the `inference` directory. 

* Inference on NCI ERA5
  The code for inference on NCI ERA5 is under the `inference_nci` directory. 

Inference on NCI ERA5
---------------------

The inference algorithm on NCI ERA5 works as follows:

```

INPUTS:  start_time
         end_time
         prediction_length
         initial conditions (i.e. raw ERA5 data)

OUTPUTS: predictions in zarr format

let t = start_time

while t <= end_time:
  load initial conditions at t to initialize FourCastNet

  forecast from t+1 to t+1 + prediction_length

  t += t+1 + prediction_length

```

The inference code is tested to support both CPU and GPU. With GPU support, we use CUDA 11.7 which can
be loaded via `module load cuda/11.7.0` on Gadi.

A demonstration to run the inference script `inference_nci/inference.py` is `test_inference.sh`.

* To run `test_inference.sh`, one needs to modify `checkpoint_dir` and `stats_dir` to point to the right
  location.

* With a short prediction length specified in `test_inference.sh`, one can run the script directly from
  Gadi login node to get quick results in a few minutes.
