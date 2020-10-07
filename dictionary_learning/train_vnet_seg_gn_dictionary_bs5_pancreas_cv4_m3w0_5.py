#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:30:26 2018

@author: xiaoyangchen
"""

from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
from keras.utils import to_categorical

from Vnet3d import vnet

batchSize = 5
numofclasses = 2
input_channel = 2
base_size = 64

lambda_ce = 1
lambda_dis = 0.5

start_epoch = 0
num_epochs = 40

samplesPerImageTrain = 200
samplesPerImageVal = 300
numTrainImages = 61
numValImages = 21

iterations = samplesPerImageTrain * numTrainImages // batchSize
val_iterations = samplesPerImageVal * numValImages // batchSize

model = vnet(input_channel, base_size, numofclasses, batch_size=batchSize, lambda_dis=lambda_dis, lambda_ce=lambda_ce)
#model.summary()
#assert 0

def train_generator(batch_size=1, numClasses=2, numImages=62, numSamples=200):
	# the following parameters need to be assigned values before training
    batch_size = batch_size # very important
    
    size_ = 64
    patch_size = np.array([size_, size_, size_]) # very important
    num_of_downpooling = 4 # very important
    #patch_stride_regulator = np.array([4, 4, 4]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    #stride = patch_size/patch_stride_regulator
    
    num_classes = numClasses
    data_image_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/images/'
    data_seg_gt_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/label/'
    
    subject_list = list(range(1, 21)) + list(range(21, 41)) + list(range(41, 62))
    assert len(subject_list) == numImages
    label = 0
    
    while True:
        np.random.shuffle(subject_list)
        for j in range(len(subject_list)):
            
            subject_index = subject_list[j]
            
            image = sitk.ReadImage(data_image_path + 'image{:04d}.nii.gz'.format(subject_index))
            image = sitk.GetArrayFromImage(image)
            image = image / np.max(image)
            
            segmentation_gt = sitk.ReadImage(data_seg_gt_path + 'label{:04d}.nii.gz'.format(subject_index))
            segmentation_gt = sitk.GetArrayFromImage(segmentation_gt)
            unique_ids = np.unique(segmentation_gt)
            nonzero_unique_ids = unique_ids[np.where(unique_ids > 0)]
            assert np.all([x < num_classes for x in nonzero_unique_ids])
            assert nonzero_unique_ids is not np.array([])
            
            segmentation_gt = np.transpose(to_categorical(segmentation_gt, num_classes).reshape(list(segmentation_gt.shape + (num_classes,))), [3, 0, 1, 2])
            
            image_size = image.shape
            seg_gt_size = segmentation_gt.shape
            assert seg_gt_size[1] == image_size[0] and seg_gt_size[2] == image_size[1] and seg_gt_size[3] == image_size[2]
            
            numSamples = numSamples
            vertex = np.zeros([numSamples, 3, 6]) ## in the order of (1000, 3, z1, z2, y1, y2, x1, x2)
            shapedata = vertex.shape
            
            patch_index = 0 ## update by 1 after generating a patch
            margin = 10
            assert margin <= size_//2
            
            for label_i in nonzero_unique_ids:
                m = segmentation_gt[label_i]
                # Bounding box.
                depth_indicies = np.where(m == 1)[0]
                height_indicies = np.where(m == 1)[1]
                width_indicies = np.where(m == 1)[2]
                
                z1, z2 = np.min(depth_indicies), np.max(depth_indicies)
                y1, y2 = np.min(height_indicies), np.max(height_indicies)
                x1, x2 = np.min(width_indicies), np.max(width_indicies)
                
                for ii in range(int(0.8*numSamples/len(nonzero_unique_ids))):
                    center_z = np.random.randint(z1-margin, z2+margin+1)
                    center_y = np.random.randint(y1-margin, y2+margin+1)
                    center_x = np.random.randint(x1-margin, x2+margin+1)
                    
                    vertex[patch_index][0] = np.array([center_z-int(patch_size[0]//2), center_z+int(patch_size[0]//2),
                                                       center_y-int(patch_size[1]//2), center_y+int(patch_size[1]//2),
                                                       center_x-int(patch_size[2]//2), center_x+int(patch_size[2]//2)])

                    vertex[patch_index][1] = np.array([center_z-2*int(patch_size[0]//2), center_z+2*int(patch_size[0]//2),
                                                       center_y-2*int(patch_size[1]//2), center_y+2*int(patch_size[1]//2),
                                                       center_x-2*int(patch_size[2]//2), center_x+2*int(patch_size[2]//2)])

                    patch_index += 1
                        
            while patch_index < numSamples:

                center_z = np.random.randint(size_//2, image_size[0]-size_//2)
                center_y = np.random.randint(size_//2, image_size[1]-size_//2)
                center_x = np.random.randint(size_//2, image_size[2]-size_//2)
                
                vertex[patch_index][0] = np.array([center_z-int(patch_size[0]//2), center_z+int(patch_size[0]//2),
                                                   center_y-int(patch_size[1]//2), center_y+int(patch_size[1]//2),
                                                   center_x-int(patch_size[2]//2), center_x+int(patch_size[2]//2)])
                
                vertex[patch_index][1] = np.array([center_z-2*int(patch_size[0]//2), center_z+2*int(patch_size[0]//2),
                                                   center_y-2*int(patch_size[1]//2), center_y+2*int(patch_size[1]//2),
                                                   center_x-2*int(patch_size[2]//2), center_x+2*int(patch_size[2]//2)])
                
                patch_index += 1
            
            modulo=np.mod(shapedata[0], batch_size)
            if modulo!=0:
                num_to_add=batch_size-modulo
                inds_to_add=np.random.randint(0, shapedata[0], num_to_add) ## the return value is a ndarray
                to_add = vertex[inds_to_add]
                new_vertex = np.vstack((vertex, to_add))
            else:
                new_vertex = vertex
            
            np.random.shuffle(new_vertex)
            for i_batch in range(int(new_vertex.shape[0]/batch_size)):
                subvertex = new_vertex[i_batch*batch_size:(i_batch+1)*batch_size]
                for count in range(batch_size):
                    ## size_*size_*size_ ##
                    image_one = np.zeros([size_, size_, size_], dtype=np.float32)
                    seg_gt_one = np.zeros([num_classes, size_, size_, size_], dtype=np.float32)
                    seg_gt_one[0] = np.ones([size_, size_, size_], dtype=np.float32) ## I made a huge mistake here ##
                    
                    copy_from, copy_to = corrected_crop(subvertex[count][0], np.array(list(image_size)))

                    cf_z_lower_bound = int(copy_from[0])
                    if copy_from[1] is not None:
                        cf_z_higher_bound = int(copy_from[1])
                    else:
                        cf_z_higher_bound = None
                    
                    cf_y_lower_bound = int(copy_from[2])
                    if copy_from[3] is not None:
                        cf_y_higher_bound = int(copy_from[3])
                    else:
                        cf_y_higher_bound = None
                    
                    cf_x_lower_bound = int(copy_from[4])
                    if copy_from[5] is not None:
                        cf_x_higher_bound = int(copy_from[5])
                    else:
                        cf_x_higher_bound = None

                    image_one[int(copy_to[0]):copy_to[1],
                              int(copy_to[2]):copy_to[3],
                              int(copy_to[4]):copy_to[5]] = \
                              image[cf_z_lower_bound:cf_z_higher_bound,
                                    cf_y_lower_bound:cf_y_higher_bound,
                                    cf_x_lower_bound:cf_x_higher_bound]

                    seg_gt_one[:,
                               int(copy_to[0]):copy_to[1],
                               int(copy_to[2]):copy_to[3],
                               int(copy_to[4]):copy_to[5]] = \
                               segmentation_gt[:,
                                                cf_z_lower_bound:cf_z_higher_bound,
                                                cf_y_lower_bound:cf_y_higher_bound,
                                                cf_x_lower_bound:cf_x_higher_bound]

                    image_one = np.expand_dims(image_one, axis=0)

                    ## (2*size_)*(2*size_)*(2*size_) ##
                    image_two = np.zeros([2*size_, 2*size_, 2*size_], dtype=np.float32)

                    copy_from, copy_to = corrected_crop(subvertex[count][1], np.array(list(image_size)))
                    
                    cf_z_lower_bound = int(copy_from[0])
                    if copy_from[1] is not None:
                        cf_z_higher_bound = int(copy_from[1])
                    else:
                        cf_z_higher_bound = None
                    
                    cf_y_lower_bound = int(copy_from[2])
                    if copy_from[3] is not None:
                        cf_y_higher_bound = int(copy_from[3])
                    else:
                        cf_y_higher_bound = None
                    
                    cf_x_lower_bound = int(copy_from[4])
                    if copy_from[5] is not None:
                        cf_x_higher_bound = int(copy_from[5])
                    else:
                        cf_x_higher_bound = None
                    
                    image_two[(copy_to[0]):(copy_to[1]),
                              (copy_to[2]):(copy_to[3]),
                              (copy_to[4]):(copy_to[5])] = \
                              image[cf_z_lower_bound:cf_z_higher_bound,
                                    cf_y_lower_bound:cf_y_higher_bound,
                                    cf_x_lower_bound:cf_x_higher_bound]

                    image_two = np.expand_dims(image_two, axis=0)
                    
                    ## output batch ##
                    image_1 = np.expand_dims(image_one, axis=0)
                    image_2 = np.expand_dims(image_two, axis=0)
                    seg_gt_one = np.expand_dims(seg_gt_one, axis=0)

                    if label == 0:
                        Img_1 = image_1
                        Img_2 = image_2
                        seg_gt = seg_gt_one
                        label += 1
                    else:
                        Img_1 = np.vstack((Img_1, image_1))
                        Img_2 = np.vstack((Img_2, image_2))
                        seg_gt = np.vstack((seg_gt, seg_gt_one))
                        label += 1
                    
                    if np.remainder(label, batch_size)==0:
                        #mask = np.ones([batch_size, num_classes, size_, size_, size_], dtype=np.int32)
                        yield ([Img_1, Img_2, seg_gt], [])
                        label = 0

def corrected_crop(array, image_size):
    array_ = array.copy()
    image_size_ = image_size.copy()
    
    copy_from = [0, 0, 0, 0, 0, 0] #np.zeros([6,], dtype=np.int32)
    copy_to = [0, 0, 0, 0, 0, 0] #np.zeros([6,], dtype=np.int32)
    ## 0 ##
    if array[0] < 0:
        copy_from[0] = 0
        copy_to[0] = int(abs(array_[0]))
    else:
        copy_from[0] = int(array_[0])
        copy_to[0] = 0
    ## 1 ##
    if array[1] > image_size_[0]:
        copy_from[1] = None
        copy_to[1] = -int(array_[1] - image_size_[0])
    else:
        copy_from[1] = int(array_[1])
        copy_to[1] = None
    ## 2 ##
    if array[2] < 0:
        copy_from[2] = 0
        copy_to[2] = int(abs(array_[2]))
    else:
        copy_from[2] = int(array_[2])
        copy_to[2] = 0
    ## 3 ##
    if array[3] > image_size_[1]:
        copy_from[3] = None
        copy_to[3] = -int(array_[3] - image_size_[1])
    else:
        copy_from[3] = int(array_[3])
        copy_to[3] = None
    ## 4 ##
    if array[4] < 0:
        copy_from[4] = 0
        copy_to[4] = int(abs(array_[4]))
    else:
        copy_from[4] = int(array_[4])
        copy_to[4] = 0
    ## 5 ##  
    if array[5] > image_size_[2]:
        copy_from[5] = None
        copy_to[5] = -int(array_[5] - image_size_[2])
    else:
        copy_from[5] = int(array_[5])
        copy_to[5] = None

    return copy_from, copy_to

def validation_generator(batch_size=1, numClasses=2, numImages=20, numSamples=300):
	# the following parameters need to be assigned values before training
    batch_size = batch_size # very important
    
    size_ = 64
    patch_size = np.array([size_, size_, size_]) # very important
    num_of_downpooling = 4 # very important
    patch_stride_regulator = np.array([1, 1, 1]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    stride = patch_size/patch_stride_regulator
    
    num_classes = numClasses
    data_image_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/images/'
    data_seg_gt_path = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/label/'
    
    subject_list = list(range(62, 83))
    assert len(subject_list) == numImages
    
    while True:
        label = 0
        for idx in range(len(subject_list)):
            subject_index = subject_list[idx]

            image = sitk.ReadImage(data_image_path + 'image{:04d}.nii.gz'.format(subject_index))
            image = sitk.GetArrayFromImage(image)
            image = image / np.max(image)

            segmentation_gt = sitk.ReadImage(data_seg_gt_path + 'label{:04d}.nii.gz'.format(subject_index))
            segmentation_gt = sitk.GetArrayFromImage(segmentation_gt)
            unique_ids = np.unique(segmentation_gt)
            nonzero_unique_ids = unique_ids[np.where(unique_ids > 0)]
            assert nonzero_unique_ids is not np.array([])
            segmentation_gt = np.transpose(to_categorical(segmentation_gt, num_classes).reshape(list(segmentation_gt.shape + (num_classes,))), [3, 0, 1, 2])

            image_size = np.array(np.shape(image), dtype=np.int16)

            expanded_image_size = (np.ceil(image_size/(1.0*stride))*stride).astype(np.int16)
            
            expanded_image = np.zeros(expanded_image_size, dtype=np.float32)
            expanded_image[0:image_size[0], 0:image_size[1], 0:image_size[2]] = image
        
            expanded_seg_gt = np.zeros([num_classes, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
            expanded_seg_gt[0] = np.ones([expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32) ## I really made a huge mistake here ##
            expanded_seg_gt[:, 0:image_size[0], 0:image_size[1], 0:image_size[2]] = segmentation_gt

            numSamples = 300
            margin = 0
            center = np.zeros([numSamples, 3])
            
            # foreground samples #
            patch_index = 0
            for label_i in nonzero_unique_ids:
                m = expanded_seg_gt[label_i]
                # Bounding box.
                depth_indicies = np.where(m == 1)[0]
                height_indicies = np.where(m == 1)[1]
                width_indicies = np.where(m == 1)[2]
                
                z1, z2 = np.min(depth_indicies), np.max(depth_indicies)
                y1, y2 = np.min(height_indicies), np.max(height_indicies)
                x1, x2 = np.min(width_indicies), np.max(width_indicies)
                
                for ii in range(int(0.8*numSamples/len(nonzero_unique_ids))):
                    center_z = np.random.randint(z1-margin, z2+margin+1)
                    center_y = np.random.randint(y1-margin, y2+margin+1)
                    center_x = np.random.randint(x1-margin, x2+margin+1)
                    
                    center[patch_index] = np.array([center_z, center_y, center_x])

                    patch_index += 1

            # Samples selected from the whole image region #
            while patch_index < numSamples:

                center_z = np.random.randint(size_//2, image_size[0]-size_//2)
                center_y = np.random.randint(size_//2, image_size[1]-size_//2)
                center_x = np.random.randint(size_//2, image_size[2]-size_//2)
                
                center[patch_index] = np.array([center_z, center_y, center_x])
                
                patch_index += 1

            modulo=np.mod(numSamples, batch_size)
            
            if modulo!=0:
                num_to_add=batch_size-modulo
                inds_to_add=np.random.randint(0, numSamples, num_to_add) ## the return value is a ndarray
                to_add = center[inds_to_add]
                new_center = np.vstack((center, to_add))
            else:
                new_center = center
            
            np.random.shuffle(new_center)

            for i_batch in range(int(new_center.shape[0]/batch_size)):
                subvertex = new_center[i_batch*batch_size:(i_batch+1)*batch_size]
                for count in range(batch_size):
                    ## size_*size_*size_ ##
                    image_one = np.zeros([size_, size_, size_], dtype=np.float32)
                    seg_gt_one = np.zeros([num_classes, size_, size_, size_], dtype=np.float32)
                    seg_gt_one[0] = np.ones([size_, size_, size_], dtype=np.float32) ## To make sure voxels in the padded part are assaigned with label 0 ##
                    
                    z_lower_bound = int(subvertex[count][0] - patch_size[0]//2)
                    z_higher_bound = int(subvertex[count][0] + patch_size[0]//2)
                    y_lower_bound = int(subvertex[count][1] - patch_size[1]//2)
                    y_higher_bound = int(subvertex[count][1] + patch_size[1]//2)
                    x_lower_bound = int(subvertex[count][2] - patch_size[2]//2)
                    x_higher_bound = int(subvertex[count][2] + patch_size[2]//2)
                    
                    virgin_range = np.array([z_lower_bound, z_higher_bound, y_lower_bound, y_higher_bound, x_lower_bound, x_higher_bound])
                    copy_from, copy_to = corrected_crop(virgin_range, expanded_image_size)
    
                    cf_z_lower_bound = int(copy_from[0])
                    if copy_from[1] is not None:
                        cf_z_higher_bound = int(copy_from[1])
                    else:
                        cf_z_higher_bound = None
                    
                    cf_y_lower_bound = int(copy_from[2])
                    if copy_from[3] is not None:
                        cf_y_higher_bound = int(copy_from[3])
                    else:
                        cf_y_higher_bound = None
                    
                    cf_x_lower_bound = int(copy_from[4])
                    if copy_from[5] is not None:
                        cf_x_higher_bound = int(copy_from[5])
                    else:
                        cf_x_higher_bound = None
                    
                    image_one[int(copy_to[0]):copy_to[1],
                              int(copy_to[2]):copy_to[3],
                              int(copy_to[4]):copy_to[5]] = \
                              expanded_image[cf_z_lower_bound:cf_z_higher_bound,
                                             cf_y_lower_bound:cf_y_higher_bound,
                                             cf_x_lower_bound:cf_x_higher_bound]

                    seg_gt_one[:,
                               int(copy_to[0]):copy_to[1],
                               int(copy_to[2]):copy_to[3],
                               int(copy_to[4]):copy_to[5]] = \
                               expanded_seg_gt[:,
                                                cf_z_lower_bound:cf_z_higher_bound,
                                                cf_y_lower_bound:cf_y_higher_bound,
                                                cf_x_lower_bound:cf_x_higher_bound]
    
                    image_one = np.expand_dims(image_one, axis=0)
                    
                    ## (2*size_)*(2*size_)*(2*size_) ##
                    image_two = np.zeros([2*size_, 2*size_, 2*size_], dtype=np.float32)
    
                    z_lower_bound = int(subvertex[count][0] - patch_size[0])
                    z_higher_bound = int(subvertex[count][0] + patch_size[0])
                    y_lower_bound = int(subvertex[count][1] - patch_size[1])
                    y_higher_bound = int(subvertex[count][1] + patch_size[1])
                    x_lower_bound = int(subvertex[count][2] - patch_size[2])
                    x_higher_bound = int(subvertex[count][2] + patch_size[2])
    
                    virgin_range = np.array([z_lower_bound, z_higher_bound, y_lower_bound, y_higher_bound, x_lower_bound, x_higher_bound])
                    copy_from, copy_to = corrected_crop(virgin_range, expanded_image_size)
                    
                    cf_z_lower_bound = int(copy_from[0])
                    if copy_from[1] is not None:
                        cf_z_higher_bound = int(copy_from[1])
                    else:
                        cf_z_higher_bound = None
                    
                    cf_y_lower_bound = int(copy_from[2])
                    if copy_from[3] is not None:
                        cf_y_higher_bound = int(copy_from[3])
                    else:
                        cf_y_higher_bound = None
                    
                    cf_x_lower_bound = int(copy_from[4])
                    if copy_from[5] is not None:
                        cf_x_higher_bound = int(copy_from[5])
                    else:
                        cf_x_higher_bound = None
                    
                    image_two[int(copy_to[0]):copy_to[1],
                              int(copy_to[2]):copy_to[3],
                              int(copy_to[4]):copy_to[5]] = \
                              expanded_image[cf_z_lower_bound:cf_z_higher_bound,
                                             cf_y_lower_bound:cf_y_higher_bound,
                                             cf_x_lower_bound:cf_x_higher_bound]
    
                    image_two = np.expand_dims(image_two, axis=0)
    
                    ## output batch ##
                    image_1 = np.expand_dims(image_one, axis=0)
                    image_2 = np.expand_dims(image_two, axis=0)
                    seg_gt_one = np.expand_dims(seg_gt_one, axis=0)
    
                    if label == 0:
                        Img_1 = image_1
                        Img_2 = image_2
                        seg_gt = seg_gt_one
                        label += 1
                    else:
                        Img_1 = np.vstack((Img_1, image_1))
                        Img_2 = np.vstack((Img_2, image_2))
                        seg_gt = np.vstack((seg_gt, seg_gt_one))
                        label += 1
                    
                    if np.remainder(label, batch_size)==0:
                        #mask = np.ones([batch_size, num_classes, size_, size_, size_], dtype=np.int32)
                        yield ([Img_1, Img_2, seg_gt], [])
                        label = 0

def lr_schedule(epoch):
    if epoch <= 1:
        lr = 0.0002
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 3:
        lr = 0.0001
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 8:
        lr = 5e-5
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    elif epoch <= 12:
        lr = 2e-5
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr
    else:
        lr = 1e-5
        print('Learning rate of epoch {0} is {1}'.format(epoch, lr))
        return lr

def Train():
    train_gen = train_generator(batch_size=batchSize, numClasses=numofclasses, numImages=numTrainImages, numSamples=samplesPerImageTrain)
    val_gen = validation_generator(batch_size=batchSize, numClasses=numofclasses, numImages=numValImages, numSamples=samplesPerImageVal)
    
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    
    # start from checkpoints
    if start_epoch > 0:
        model.load_weights('./checkpoints/supervised_model1_best.h5')

    best_full_loss = 1000000000.0
    for epoch in range(start_epoch, num_epochs):
        #K.set_value(model1.optimizer.lr, Segmenter_lr_schedule(epoch+1))
        losses = []
        losses_dis = []
        losses_seg = []
        for i_iter in range(iterations):
            # Train with labeled data
            [image1, image2, batch_label], _ = next(train_gen)
            
            loss, loss_dis, loss_seg = model.train_on_batch([image1, image2, batch_label], [])
            losses.append(loss)
            losses_dis.append(loss_dis)
            losses_seg.append(loss_seg)
            
            if np.mod(i_iter+1, 200//batchSize) == 0:
                print('Epoch:{0:3d}, iter = {1:5d}, loss = {2:.4f}, loss_dis = {3:.4f}, losses_seg = {4:.4f}'.format(epoch+1, i_iter+1, np.mean(np.array(losses)), np.mean(np.array(losses_dis)), np.mean(np.array(losses_seg))))
                losses = []
                losses_dis = []
                losses_seg = []
        
        # Validation
        loss_sum = 0
        for vi_iter in range(val_iterations):
            [image1, image2, batch_label], _ = next(val_gen)
            val_loss, val_loss_dis, val_loss_seg = model.test_on_batch([image1, image2, batch_label], [])
            loss_sum += val_loss_seg/val_iterations
        
        current_loss = loss_sum
        print("Validation loss is {0}".format(current_loss))
        if current_loss < best_full_loss:
            best_full_loss = current_loss
            model.save_weights('./checkpoints/supervised_model1_best.h5')

        if current_loss < 0.046:
            model.save_weights('./checkpoints/supervised_model1_epoch{0}_avg{1:5f}.h5'.format(epoch+1, current_loss))
        
if __name__ == '__main__':

    Train()

