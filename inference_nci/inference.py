#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation
#Edison Guo - National Computational Infrastructure

import os
import sys

import time
import numpy as np
import argparse
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
from collections import OrderedDict
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import zarr

img_shape_x = 720
img_shape_y = 1440

def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_model(model, device, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname, map_location=device)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def downsample(x, scale=0.125):
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')

def setup(params):
    device = get_device()
    logging.info(f'device: {device}')

    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    if params.log_to_screen:
        logging.info('Loading backbone model checkpoint from {}'.format(params['backbone_checkpoint_path']))
        logging.info('Loading precip model checkpoint from {}'.format(params['precip_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    
    if params["orography"]:
      params['N_in_channels'] = n_in_channels + 1
    else:
      params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[0, out_channels] # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    backbone_model = AFNONet(params).to(device) 
    checkpoint_file  = params['backbone_checkpoint_path']
    backbone_model = load_model(backbone_model, device, params, checkpoint_file).to(device)

    params['N_out_channels'] = 1
    model = AFNONet(params).to(device) 
    precip_model = PrecipNet(params, model).to(device) 
    checkpoint_file  = params['precip_checkpoint_path']
    precip_model = load_model(precip_model, device, params, checkpoint_file).to(device)

    return backbone_model, precip_model

def autoregressive_inference(params, valid_data, backbone_model, precip_model): 
    #initialize global variables
    device = get_device()
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    seq_pred = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    precip_seq_pred = torch.zeros((prediction_length, 1, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    # standardize
    valid_data = (valid_data - means)/stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    std = torch.as_tensor(stds[:,0,0]).to(device, dtype=torch.float)

    orography = params.orography
    orography_path = params.orography_path
    if orography:
      import h5py
      orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:img_shape_x], axis = 0), axis = 0)).to(device, dtype = torch.float)
      logging.info("orography loaded; shape:{}".format(orog.shape))

    for i_ens in range(params.ensemble_size):
      for i in range(prediction_length): 
        if i==0: #start of sequence
          first = valid_data
          if params.ensemble_size > 1:
            first = gaussian_perturb(first, level=params.ensemble_noise_std, device=device) # perturb the ic

          with torch.inference_mode():
            if orography:
              future_pred = backbone_model(torch.cat((first, orog), axis=1))
            else:
              future_pred = backbone_model(first)
            precip_future_pred = precip_model(future_pred)

        else:
          with torch.inference_mode():
            if orography:
              future_pred = backbone_model(torch.cat((future_pred, orog), axis=1)) #autoregressive step
            else:
              future_pred = backbone_model(future_pred) #autoregressive step
            precip_future_pred = precip_model(future_pred)

        if i_ens == 0:
          seq_pred[i] = future_pred
          precip_seq_pred[i] = precip_future_pred
        else:
          seq_pred[i] += future_pred[0, ...]
          precip_seq_pred[i] += precip_future_pred[0, ...]

    if params.ensemble_size > 1:
      seq_pred[i] /= float(params.ensemble_size)
      precip_seq_pred[i] /= float(params.ensemble_size)

    seq_pred = seq_pred.cpu().numpy() * stds + means
    precip_seq_pred = unlog_tp_torch(precip_seq_pred).cpu().squeeze(1).numpy()

    return seq_pred, precip_seq_pred

def parse_datetime(ts):
    time_formats = [
        '%Y-%m-%d', '%Y-%m-%dT%H', '%Y-%m-%dT%H:%M:%S',
        '%Y-%-m-%-d', '%Y-%-m-%-dT%-H', '%Y-%-m-%-dT%-H:%-M:%-S']
    for tf in time_formats:
        try:
            dt = datetime.strptime(ts, tf)
            normalized_dt = datetime(dt.year, dt.month, dt.day, hour=dt.hour // 6 * 6)
            return normalized_dt
        except Exception as exc:
            pass

    raise ValueError(f'invalid datetime format: {ts}. Supported datetime formats are: {time_formats}')
    
def get_prediction_times(start_datetime, end_datetime, prediction_length):
    prediction_times = []

    curr_time = start_datetime
    while curr_time <= end_datetime:
        pred_times = []
        for ip in range(prediction_length+1):
            pred_time = curr_time + relativedelta(hours=ip*6)
            pred_times.append(pred_time)

            if ip == prediction_length:
                next_time = pred_time

        prediction_times.append(pred_times)
        curr_time = next_time

    return prediction_times

if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__)) 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", type=str, required=True)
    parser.add_argument("--end_time", default='', type=str)
    parser.add_argument("--yaml_config", default='', type=str)
    parser.add_argument("--config", default='afno_backbone', type=str)
    parser.add_argument("--use_daily_climatology", action='store_true')
    parser.add_argument("--output_path", default=None, type = str, help = 'Path to store inference outputs')
    parser.add_argument("--interp", default=0, type=float)
    parser.add_argument("--checkpoint_dir", default=None, type=str, help = 'Path to model checkpoint directory')
    parser.add_argument("--stats_dir", default=None, type=str, help = 'Path to stats root directory')
    parser.add_argument("--prediction_length", default=40, type=int, help = 'prediction length')
    parser.add_argument("--ensemble_size", default=1, type=int, help = 'ensemble size')
    parser.add_argument("--ensemble_noise_std", default=1e-6, type=float, help = 'ensemble noise standard deviation')
    args = parser.parse_args()

    src_root = os.path.dirname(curr_dir)
    for d in [src_root, curr_dir]:
        if d not in sys.path:
            sys.path.append(d)

    from utils import logging_utils
    from utils.weighted_acc_rmse import unlog_tp_torch
    from utils.YParams import YParams
    import data_loader as dl
    from networks.afnonet import AFNONet, PrecipNet

    logging_utils.config_logger()

    if len(args.yaml_config) == 0:
        args.yaml_config = os.path.join(src_root, 'config/AFNO.yaml')

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    params['interp'] = args.interp
    params['use_daily_climatology'] = args.use_daily_climatology
    params['global_batch_size'] = params.batch_size

    if args.prediction_length > 0:
        params['prediction_length'] = args.prediction_length

    prediction_length = params['prediction_length']

    params['ensemble_size'] = args.ensemble_size
    if params['ensemble_size'] < 1:
        params['ensemble_size'] = 1

    if params['ensemble_size'] > 1:
        torch.manual_seed(0)
        np.random.seed(0)

    params['ensemble_noise_std'] = args.ensemble_noise_std

    start_datetime = parse_datetime(args.start_time)
    if len(args.end_time) > 0:
        end_datetime = parse_datetime(args.end_time)
    else:
        end_datetime = start_datetime

    if args.stats_dir is not None:
        params['time_means_path'] = os.path.join(args.stats_dir, 'time_means.npy')
        params['global_means_path'] = os.path.join(args.stats_dir, 'global_means.npy')
        params['global_stds_path'] = os.path.join(args.stats_dir, 'global_stds.npy')

    output_root = os.path.dirname(args.output_path)
    if not os.path.exists(output_root):
      os.makedirs(output_root)

    params['backbone_checkpoint_path'] = os.path.join(args.checkpoint_dir, 'backbone.ckpt') 
    params['precip_checkpoint_path'] = os.path.join(args.checkpoint_dir, 'precip.ckpt') 
    params['resuming'] = False
    params['local_rank'] = 0

    params.log()

    backbone_model, precip_model = setup(params)

    prediction_times = get_prediction_times(start_datetime, end_datetime, prediction_length)
    n_ics = len(prediction_times)
    n_times = n_ics * prediction_length

    logging.info("Inference for {} initial conditions, ensemble_size={}".format(n_ics, params['ensemble_size']))

    ds_root = zarr.open(args.output_path, 'w')
    num_channels = 20
    ds_vars = []
    for ic in range(num_channels):
        var_name = dl.channel_to_var(ic)
        ds_var = ds_root.create(var_name,
            shape=(n_times, img_shape_x, img_shape_y),
            chunks=(1, img_shape_x, img_shape_y),
            fill_value=np.nan,
            dtype=np.float32)
        ds_var.attrs['_ARRAY_DIMENSIONS'] = ['time', 'x', 'y']
        ds_vars.append(ds_var)

    tp_var = ds_root.create('tp',
        shape=(n_times, img_shape_x, img_shape_y),
        chunks=(1, img_shape_x, img_shape_y),
        fill_value=np.nan,
        dtype=np.float32)
    tp_var.attrs['_ARRAY_DIMENSIONS'] = ['time', 'x', 'y']
    ds_vars.append(tp_var)

    coord_time = ds_root.create('time',
        shape=(n_times, ),
        chunks=(n_times, ),
        dtype='datetime64[ns]')
    coord_time.attrs['_ARRAY_DIMENSIONS'] = ['time', ]

    pred_times = []
    for pred_t in prediction_times:
        for t in pred_t[1:]:
            pred_times.append(np.datetime64(t))
    coord_time[:] = np.array(pred_times, dtype='datetime64[ns]')

    in_channels = np.array(params.in_channels)

    for ic in range(n_ics):
        query_time = prediction_times[ic][0]
        logging.info(f'initial condition({ic+1}/{n_ics}): {query_time}')

        init_vals = dl.get_data(query_time)
        init_vals = init_vals[:, in_channels, 0:img_shape_x]

        seq_pred, precip_seq_pred = autoregressive_inference(params, init_vals, backbone_model, precip_model)

        t_start = ic * prediction_length
        t_end = t_start + prediction_length
        for c_idx in range(num_channels):
            ds_vars[c_idx][t_start:t_end, ...] = seq_pred[:, c_idx, ...]

        ds_vars[-1][t_start:t_end, ...] = precip_seq_pred
