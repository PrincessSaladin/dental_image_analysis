import argparse
import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from segmentation3d.dataloader.image_tools import save_intermediate_results
from segmentation3d.dataloader.sampler import EpochConcateSampler
from segmentation3d.loss.focal_loss import FocalLoss
from segmentation3d.loss.multi_dice_loss import MultiDiceLoss
from segmentation3d.utils.file_io import load_config, setup_logger
from jsd.dataloader.dataset import RegressionDataset
from jsd.utils.model_io import load_checkpoint, save_checkpoint


def train(config_file):
    """ Medical image segmentation training engine
    :param config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = load_config(config_file)

    # clean the existing folder if training from scratch
    if os.path.isdir(cfg.general.save_dir):
        if cfg.general.resume_epoch < 0:
            shutil.rmtree(cfg.general.save_dir)
            os.makedirs(cfg.general.save_dir)
            shutil.copy(config_file, os.path.join(cfg.general.save_dir, 'train_config.py'))
        else:
            shutil.copy(config_file, os.path.join(cfg.general.save_dir, 'train_config.py'))
    else:
        os.makedirs(cfg.general.save_dir)
        shutil.copy(config_file, os.path.join(cfg.general.save_dir, 'train_config.py'))

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = setup_logger(log_file, 'seg3d')

    # control randomness during training
    np.random.seed(cfg.general.seed)
    torch.manual_seed(cfg.general.seed)
    if cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(cfg.general.seed)

    # dataset
    dataset = RegressionDataset(
                imlist_file=cfg.general.imseg_list,
                num_classes=cfg.dataset.num_classes,
                landmark_folder=cfg.general.landmark_folder,
                num_landmarks=cfg.dataset.num_landmarks,
                crop_spacing=cfg.dataset.crop_spacing,
                crop_size=cfg.dataset.crop_size,
                sampling_method=cfg.dataset.sampling_method,
                random_translation=cfg.dataset.random_translation,
                interpolation=cfg.dataset.interpolation,
                crop_normalizers=cfg.dataset.crop_normalizers)

    sampler = EpochConcateSampler(dataset, cfg.train.epochs)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batchsize,
                             num_workers=cfg.train.num_threads, pin_memory=True)

    net_module = importlib.import_module('jsd.network.' + cfg.net.name)
    net = net_module.RegressionNet(
        cfg.dataset.crop_size, dataset.num_modality(), cfg.dataset.num_landmarks * 3)
    max_stride = net.max_stride()
    net_module.parameters_kaiming_init(net)
    if cfg.general.num_gpus > 0:
        net = nn.parallel.DataParallel(net, device_ids=list(range(cfg.general.num_gpus)))
        net = net.cuda()

    assert np.all(np.array(cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    # training optimizer
    opt = optim.Adam(net.parameters(), lr=cfg.train.lr, betas=cfg.train.betas)

    # load checkpoint if resume epoch > 0
    if cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(cfg.general.resume_epoch, net, opt, cfg.general.save_dir)
    else:
        last_save_epoch, batch_start = 0, 0

    if cfg.loss.name == 'Focal':
        # reuse focal loss if exists
        loss_func = FocalLoss(class_num=cfg.dataset.num_classes, alpha=cfg.loss.obj_weight, gamma=cfg.loss.focal_gamma,
                              use_gpu=cfg.general.num_gpus > 0)
    elif cfg.loss.name == 'Dice':
        loss_func = MultiDiceLoss(weights=cfg.loss.obj_weight, num_class=cfg.dataset.num_classes,
                                  use_gpu=cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    regression_loss_func = nn.L1Loss()

    writer = SummaryWriter(os.path.join(cfg.general.save_dir, 'tensorboard'))

    batch_idx = batch_start
    data_iter = iter(data_loader)

    # loop over batches
    for i in range(len(data_loader)):
        begin_t = time.time()

        crops, landmarks_label, frames, filenames = data_iter.next()

        # if cfg.general.num_gpus > 0:
        #     crops, masks = crops.cuda(), masks.cuda()
        if cfg.general.num_gpus > 0:
            crops, landmarks_label = crops.cuda(), landmarks_label.cuda()

        # clear previous gradients
        opt.zero_grad()
        
        landmarks_pred = net(crops)
        train_loss = regression_loss_func(landmarks_pred, landmarks_label)
        train_loss.backward()

        # update weights
        opt.step()

        # save training crops for visualization
        if cfg.debug.save_inputs:
            batch_size = crops.size(0)
            save_intermediate_results(list(range(batch_size)), crops, None, None, frames, filenames,
                                      os.path.join(cfg.general.save_dir, 'batch_{}'.format(i)))

        epoch_idx = batch_idx * cfg.train.batchsize // len(dataset)
        batch_idx += 1
        batch_duration = time.time() - begin_t
        sample_duration = batch_duration * 1.0 / cfg.train.batchsize

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch_idx, batch_idx, train_loss.item(), sample_duration)
        logger.info(msg)

        # save checkpoint
        if epoch_idx != 0 and (epoch_idx % cfg.train.save_epochs == 0):
            if last_save_epoch != epoch_idx:
                save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, config_file, max_stride, dataset.num_modality())
                last_save_epoch = epoch_idx

        writer.add_scalar('Train/Loss', train_loss.item(), batch_idx)

    writer.close()


def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,5,6,7'

    long_description = "Training engine for 3d medical image segmentation"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='./config/train_config.py',
                        help='configure file for medical image segmentation training.')
    args = parser.parse_args()

    train(args.input)


if __name__ == '__main__':
    main()
