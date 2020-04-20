from __future__ import print_function
import argparse
import importlib
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from md_detection3d.dataset.landmark.landmark_detection_dataset import LandmarkDetectionDataset
from md_detection3d.dataset.landmark.landmark_detection_dataset import read_image_list
from md_detection3d.dataset.landmark.landmark_detection_dataset import read_label_list
from md.mdpytorch.loss.focal_loss import FocalLoss
from md.mdpytorch.utils.train_tools import EpochConcateSampler
from md.utils.python.plotly_tools import plot_loss
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from md_detection3d.utils.vdnet_helpers import LoadConfig
from md_detection3d.utils.vdnet_helpers import LoadCheckpoint
from md_detection3d.utils.vdnet_helpers import SaveCheckpoint
from md_detection3d.utils.vdnet_helpers import SetupTrainLogger

def ParseAndCheckArguments():
  parser = argparse.ArgumentParser()

  default_config = os.path.join(os.getcwd(), 'config',
                                'mi_aorta_detection_config.py')

  print("default config = {0}".format(default_config))

  parser.add_argument('--config', type=str, default = default_config,
                      help='Folder containing the detection training config file.')

  args = parser.parse_args()

  invalid_arguments = False
  if not args.config:
    print("Please specify the configuration file.")
    invalid_arguments = True
  elif not os.path.isfile(args.config):
    print("The specified config: {0} does not exist!".format(args.config))
    invalid_arguments = True

  if invalid_arguments:
    raise ValueError("Invalid input arguments!")

  return args

def Train(config_file):
  cfg = LoadConfig(config_file)

  # convert to absolute path since cfg use relative path
  cfg.general.label_dir = os.path.join(cfg.general.root_dir, cfg.general.label_dir)
  assert os.path.isdir(cfg.general.label_dir)

  cfg.general.image_dir = os.path.join(cfg.general.root_dir, cfg.general.image_dir)
  assert os.path.isdir(cfg.general.image_dir)

  cfg.general.save_dir = os.path.join(cfg.general.root_dir, cfg.general.save_dir)
  if not os.path.isdir(cfg.general.save_dir):
    os.makedirs(cfg.general.save_dir)

  # copy config file
  config_file_to_save = os.path.join(cfg.general.save_dir, "config.py")
  if config_file != config_file_to_save:
    shutil.copy(config_file, os.path.join(cfg.general.save_dir, "config.py"))

  # control randomness during training
  np.random.seed(cfg.general.seed)
  torch.manual_seed(cfg.general.seed)
  torch.cuda.manual_seed(cfg.general.seed)

  # clean up the existing folder if not resume training
  if not cfg.general.resume_epoch:
    shutil.rmtree(cfg.general.save_dir)

  # enable logging
  log_file = os.path.join(cfg.general.save_dir, 'logging', 'train_log.txt')
  logger = SetupTrainLogger(log_file)

  # enable cudnn
  cudnn.benchmark = True
  if not torch.cuda.is_available():
    raise EnvironmentError('CUDA is not available! Please check nvidia driver!')

  # create dataset and dataloader
  label_list = read_label_list(cfg.general.label_dir,
                               cfg.general.label_list_files)
  image_list = read_image_list(cfg.general.label_dir,
                               cfg.general.image_list_file)

  run_regression = cfg.loss.regression.lamda > 0

  dataset = LandmarkDetectionDataset(
    cfg.general.image_dir,
    image_list,
    label_list,
    cfg.dataset.voxel_spacing,
    cfg.dataset.cropping_size,
    cfg.dataset.sampling_size,
    cfg.dataset.positive_upper_bound,
    cfg.dataset.negative_lower_bound,
    cfg.dataset.num_pos_patches_per_image,
    cfg.dataset.neg_to_pos_patches_ratio,
    cfg.dataset.augmentation.on,
    cfg.dataset.augmentation.orientation_axis,
    cfg.dataset.augmentation.orientation_radian,
    cfg.dataset.normalization.mean,
    cfg.dataset.normalization.stddev,
    cfg.dataset.normalization.clip,
    run_regression)

  sampler = EpochConcateSampler(dataset, cfg.train.num_epochs)

  dataloader = DataLoader(
    dataset,
    batch_size=cfg.train.batch_size,
    sampler=sampler,
    num_workers=cfg.train.num_threads,
    pin_memory=True,
    shuffle=False)

  # define network
  net_module = importlib.import_module('md_detection3d.network.' + cfg.net.name)
  net_file, _ = os.path.splitext(net_module.__file__)
  net_file += ".py"
  shutil.copy(net_file, os.path.join(cfg.general.save_dir, "net.py"))


  in_channels = 1
  num_classes = dataset.num_classes

  net = net_module.VNet(in_channels, num_classes, run_regression)
  net_module.ApplyKaimingInit(net)
  max_stride = net.max_stride()

  gpu_ids = list(range(cfg.general.num_gpus))
  net = nn.parallel.DataParallel(net, device_ids = gpu_ids)
  net = net.cuda()

  if (cfg.dataset.cropping_size[0] % max_stride != 0) \
          or (cfg.dataset.cropping_size[1] % max_stride != 0) \
          or (cfg.dataset.cropping_size[2] % max_stride != 0):
    raise ValueError('cropping size not divisible by max_stride')

  # create optimizer.
  optimizer = optim.Adam(net.parameters(), lr=cfg.train.lr, betas=cfg.train.betas)

  # load checkpoint if resume epoch = True
  checkpoint_dir = os.path.join(cfg.general.save_dir, 'checkpoints')
  if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  if cfg.general.resume_epoch:
    batch_start = LoadCheckpoint(checkpoint_dir, net, optimizer)
  else:
    batch_start = 0

  # training buffer
  input_size = cfg.dataset.cropping_size
  dim_x = input_size[0]
  dim_y = input_size[1]
  dim_z = input_size[2]

  num_patches_per_image = cfg.dataset.num_pos_patches_per_image * (
    1 + cfg.dataset.neg_to_pos_patches_ratio)
  num_patches_per_batch = num_patches_per_image * cfg.train.batch_size
  num_input_channels = 1
  num_target_channels = 1
  num_output_channels = dataset.num_classes
  if (cfg.loss.regression.lamda > 0):
    num_target_channels = 4
    num_output_channels += 3 * (dataset.num_classes - 1)

  input = torch.FloatTensor(
    num_patches_per_batch,
    num_input_channels,
    dim_z,
    dim_y,
    dim_x)

  target = torch.FloatTensor(
    num_patches_per_batch,
    num_target_channels,
    dim_z,
    dim_y,
    dim_x)

  input = input.cuda()
  target = target.cuda()
  batch_idx = batch_start

  # define loss function
  if cfg.loss.classification.name == 'Focal':
    alpha = [None] * dataset.num_classes
    alpha[0] = 1 - cfg.loss.classification.focal_obj_alpha
    for i in range(1, num_classes):
      alpha[i] = cfg.loss.classification.focal_obj_alpha

    gamma = cfg.loss.classification.focal_gamma
    loss_func = FocalLoss(class_num=dataset.num_classes, alpha=alpha, gamma=gamma)
  else:
    raise ValueError('Unsupported loss function.')

  data_iter = iter(dataloader)
  for i in range(len(dataloader)):
    begin_t = time.time()
    crops, masks, _, _ = next(data_iter)
    crops = crops.view(-1, num_input_channels, dim_z, dim_y, dim_x)
    masks = masks.view(-1, num_target_channels, dim_z, dim_y, dim_x)

    input.resize_(crops.size())
    input.copy_(crops)

    target.resize_(masks.size())
    target.copy_(masks)

    # clear previous gradients
    optimizer.zero_grad()

    # network forward
    input_v = Variable(input)
    predictions = net(input_v)

    # both 'predictions' and 'targets' are of shape
    # [batch * patch, channels, dim_z, dim_y, dim_x]
    predictions = predictions.permute(0, 2, 3, 4, 1).contiguous()
    target = target.permute(0, 2, 3, 4, 1).contiguous()

    # reshape 'predictions' and 'targets' to [samples, channels]
    predictions = predictions.view(-1, num_output_channels)
    target = target.view(-1, num_target_channels)

    # only select those samples whose label is not -1.
    selected_sample_indices = torch.nonzero(target[:,0] != -1).squeeze()
    target = torch.index_select(target, 0, selected_sample_indices)
    selected_sample_indices_v = Variable(selected_sample_indices, requires_grad=False)
    predictions = torch.index_select(predictions, 0, selected_sample_indices_v)

    # compute training loss (ignore the regression loss)
    target_v = Variable(target, requires_grad=False)
    train_loss = loss_func(predictions[:, 0:num_classes], target_v[:, 0])

    _, predicted_labels = torch.max(predictions[:,0:num_classes].data, 1)
    ground_truth_labels = target_v[:,0].data
    train_error = float(torch.sum(predicted_labels.int() != ground_truth_labels.int())) \
                  / predicted_labels.size()[0]

    # backward propagation
    train_loss.backward()

    # update weights
    optimizer.step()

    batch_duration = time.time() - begin_t
    sample_duration = batch_duration * 1.0 / cfg.train.batch_size

    # approximate epoch_idx because len(dataset) is not necessarily divisible
    # by cfg.train.batch_size.
    epoch_idx = batch_idx * cfg.train.batch_size // len(dataset)
    # print loss per batch
    if cfg.loss.classification.name == 'Focal':
      msg = 'epoch: {}, batch: {}, floss: {:.4f}, error: {:.4f}, time: {:.4f} s/vol'
      msg = msg.format(epoch_idx, batch_idx, train_loss,
                        train_error, sample_duration)
    logger.info(msg)

    if batch_idx % cfg.train.plot_snapshot == 0:
      if cfg.loss.classification.name == 'Focal':
        floss_plot_file = os.path.join(cfg.general.save_dir, 'floss.html')
        plot_loss(log_file, floss_plot_file, name='floss', display='Focal loss')

    if epoch_idx % cfg.train.save_epochs == 0:
      SaveCheckpoint(checkpoint_dir, net, optimizer, epoch_idx, batch_idx)


    batch_idx += 1


if __name__ == '__main__':
    args = ParseAndCheckArguments()
    Train(args.config)


