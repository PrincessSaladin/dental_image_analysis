#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:20:00 2019

@author: xiaoyangchen
"""

import os
import sys
import random
import timeit
import pickle
import argparse
import scipy.misc
import numpy as np
from collections import deque

from generator import train_generator, validation_generator
from segmenter import PCANet
from discriminator import FCDiscriminator

from keras.layers import Input, Lambda
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam

import keras.backend as K
import tensorflow as tf

from losses import hybrid_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

start = timeit.default_timer()

MODEL = 'PCANet'
BASE_SIZE = 64
BATCH_SIZE = 1
ITER_SIZE = 1
SNAPSHOT_DIR = './checkpoints/'
LAMBDA_SEG = 1
LAMBDA_DIS = 0.05
LAMBDA_GT_DIS = 0.05


##################################### PARAMETERS FOR CUSTOMIZATION #################################
NUM_CLASSES = 3                                                                                    #
DATA_DIRECTORY = '/shenlab/lab_stor4/work1/xiaoyang/CBCT_Segmentation_Xiaoyang/'                   #
                                                                                                   #
EPOCHS = 50                                                                                        #
START_EPOCH = 0                                                                                    #
                                                                                                   #
NUM_OF_TRAIN_IMAGES = 20                                                                           #
NUM_SAMPLES_PER_IMAGE = 200                                                                        #
                                                                                                   #
NUM_OF_VAL_IMAGES = 5                                                                              #
NUM_VAL_SAMPLES_PER_IMAGE = 100                                                                    #
####################################################################################################

######################## DO NOT CHANGE #############################
ITERATIONS_PER_EPOCH = NUM_OF_TRAIN_IMAGES * NUM_SAMPLES_PER_IMAGE
VAL_ITERATIONS = NUM_OF_VAL_IMAGES * NUM_VAL_SAMPLES_PER_IMAGE
####################################################################

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Multiscale_Unet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : msunet")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--base-size", type=int, default=BASE_SIZE,
                        help="Base size of images sent to the network")
    parser.add_argument("--num-epochs", type=int, default=EPOCHS,
                        help="Number of epochs for training.")
    parser.add_argument("--start-epoch", type=int, default=START_EPOCH,
                        help="Starting epoch of training.")
    parser.add_argument("--num-train-images", type=int, default=NUM_OF_TRAIN_IMAGES,
                        help="Number of training images.")
    parser.add_argument("--num-samples-per-image", type=int, default=NUM_SAMPLES_PER_IMAGE,
                        help="Number of training images.")
    parser.add_argument("--num-val-images", type=int, default=NUM_OF_VAL_IMAGES,
                        help="Number of training images.")
    parser.add_argument("--num-val-samples-per-image", type=int, default=NUM_VAL_SAMPLES_PER_IMAGE,
                        help="Number of training images.")
    parser.add_argument("--iterations", type=int, default=ITERATIONS_PER_EPOCH,
                        help="Number of validation steps per epoch.")
    parser.add_argument("--val-iterations", type=int, default=VAL_ITERATIONS,
                        help="Validation steps every epoch.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory where data are stored.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_g_unlabeled for adversarial training.")
    parser.add_argument("--lambda-dis", type=float, default=LAMBDA_DIS,
                        help="lambda_d_unlabeled for adversarial training.")
    parser.add_argument("--lambda-gt-dis", type=float, default=LAMBDA_GT_DIS,
                        help="lambda_d_unlabeled for adversarial training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--use-deque", type=bool, default=True,
                        help="Whether or not to use deque to store losses.")
    parser.add_argument("--cycle", type=int, default=1,
                        help="Print loss every cycle number of iterations.")
    parser.add_argument("--capacity", type=int, default=100,
                        help="Capacity of the queue.")
    return parser.parse_args()

args = get_arguments()

def main():
    
    if args.use_deque:
        losses = deque(maxlen=args.capacity)
    else:
        losses = []

    print("Checking the consistence of parameter settings...")
    data_image_path = os.path.join(args.data_dir + 'images_nii/')
    data_seg_gt_path = os.path.join(args.data_dir + 'label_nii/')

    filenames = list(os.walk(data_image_path))[0][2]
    # filenames = [x for x in filenames if '_' not in x]
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames = sorted(filenames)
    targetnames = list(os.walk(data_seg_gt_path))[0][2]
    # targetnames = [x for x in targetnames if '_' not in x]
    targetnames = [x for x in targetnames if x.endswith('.nii.gz')]
    targetnames = sorted(targetnames)
    assert filenames == targetnames, "Found unpaired data in the data folder."

    print("INFO: {0} images used for training and validation combined.".format(len(filenames)))
    assert (args.num_val_images + args.num_train_images) == len(filenames), "Please reset NUM_OF_TRAIN_IMAGES and NUM_OF_VAL_IMAGES"

    print("Checking finished.")

    np.random.seed(2019)
    np.random.shuffle(filenames)
    train_subject_list = sorted(filenames[:NUM_OF_TRAIN_IMAGES])
    val_subject_list = sorted(filenames[-NUM_OF_VAL_IMAGES:])
    # print("train_subject_list: ", train_subject_list)
    # print("val_subject_list: ", val_subject_list)

    train_gen = train_generator(data_path=args.data_dir, subject_list=train_subject_list, num_classes=args.num_classes, samples_per_image=args.num_samples_per_image)
    val_gen = validation_generator(data_path=args.data_dir, subject_list=val_subject_list, num_classes=args.num_classes, samples_per_image=args.num_val_samples_per_image)
    
    segmenter = PCANet(numofclasses=args.num_classes)

    # Segmentation network
    input1 = Input((1, args.base_size, args.base_size, args.base_size), name='input_1')
    input2 = Input((1, 2*args.base_size, 2*args.base_size, 2*args.base_size), name='input_2')
    input3 = Input((1, 4*args.base_size, 4*args.base_size, 4*args.base_size), name='input_3')
    
    gt = Input((args.num_classes, args.base_size, args.base_size, args.base_size), name='gt')
    mask = Input((args.num_classes, args.base_size, args.base_size, args.base_size), name='mask')
    
    segmentation = segmenter(input1, input2, input3)
    seg_loss = Lambda(lambda x: hybrid_loss(*x), name="seg_loss")([gt, segmentation, mask])
    model1 = Model(inputs=[input1, input2, input3, gt, mask], outputs=[segmentation, seg_loss])
    model1.add_loss(args.lambda_seg * seg_loss)
    model1.compile(optimizer=Adam(lr=0.0005), loss=[None] * len(model1.outputs))
    
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # Start from checkpoints
    if args.start_epoch > 0:
        model1.load_weights('./checkpoints/pcanet_model1_best.h5')
    
    best_full_loss = 1000000000.0
    for epoch in range(args.start_epoch, args.num_epochs):
        K.set_value(model1.optimizer.lr, Segmenter_lr_schedule(epoch+1))
        for i_iter in range(args.iterations):
            # Train with labeled data
            [image1, image2, image3, batch_label, batch_mask], _ = next(train_gen)
            
            loss = model1.train_on_batch([image1, image2, image3, batch_label, batch_mask], [])
            losses.append(loss)
            
            if i_iter % args.cycle == 0:
                if args.use_deque:
                    print('Epoch:{0:3d}, iter = {1:4d}, loss_seg = {2:.4f}'.format(epoch+1, i_iter, np.mean(losses)))
                else:
                    print('Epoch:{0:3d}, iter = {1:4d}, loss_seg = {2:.4f}'.format(epoch+1, i_iter, np.mean(losses)))
                    losses = []
        
        # Validation
        loss_sum = 0
        for vi_iter in range(args.val_iterations):
            [image1, image2, image3, batch_label, batch_mask], _ = next(val_gen)
            val_loss_G = model1.test_on_batch([image1, image2, image3, batch_label, batch_mask], [])
            loss_sum += val_loss_G/args.val_iterations
        
        current_loss = loss_sum
        print("Validation loss is {0}".format(current_loss))
        if current_loss < best_full_loss:
            best_full_loss = current_loss
            model1.save(args.snapshot_dir + 'pcanet_model1_best.h5')

    end = timeit.default_timer()
    print(end-start,'seconds')

def Segmenter_lr_schedule(epoch):
    if epoch <= 1:
        lr = 0.0005
        print('Generator learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 4:
        lr = 0.0002
        print('Generator learning of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 10:
        lr = 1e-4
        print('Generator learning of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 20:
        lr = 1e-5
        print('Generator learning of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 40:
        lr = 5e-6
        print('Generator learning of epoch {0} is {1}'.format(epoch, lr))
        return lr
    else:
        print('Generator learning of this epoch is {0}'.format(1e-6))
        return 1e-6

if __name__ == '__main__':
    main()
