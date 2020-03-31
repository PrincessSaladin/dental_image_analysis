#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 14:37:01 2019

@author: xiaoyangchen
"""

from __future__ import print_function
import os
import h5py
import time
import random
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
import scipy.ndimage as nd
from keras.utils import to_categorical

def train_generator(data_path, subject_list, num_classes=3, samples_per_image=200, batch_size=1):
	# the following parameters need to be assigned values before training
    batch_size = batch_size # very important
    
    size_ = 64
    patch_size = np.array([size_, size_, size_]) # very important
    num_of_downpooling = 4 # very important
    patch_stride_regulator = np.array([4, 4, 4]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    stride = patch_size/patch_stride_regulator
    
    data_image_path = os.path.join(data_path + 'images_nii/')
    data_seg_gt_path = os.path.join(data_path + 'label_nii/')
    
    label = 0
    
    while True:
        np.random.shuffle(subject_list)
        for j in range(len(subject_list)):
            
            subject_name = subject_list[j]

            image = sitk.ReadImage(data_image_path + subject_name)
            image = sitk.GetArrayFromImage(image)
            image = image / np.max(image)

            segmentation_gt = sitk.ReadImage(data_seg_gt_path + subject_name)
            segmentation_gt = sitk.GetArrayFromImage(segmentation_gt)
            segmentation_gt = np.transpose(to_categorical(segmentation_gt, num_classes).reshape(list(segmentation_gt.shape + (num_classes,))), [3, 0, 1, 2])
            
            z_range_max = image.shape[0]
            ### Augmentation ###
            if image.shape[0] > 330:
                z_max = image.shape[0]
                z_range_min = random.randint(0, int(z_max*0.25))
                z_range_max = random.randint(int(z_max*0.75), z_max)
                z_range = slice(z_range_min, z_range_max)
                image = image[z_range]
                segmentation_gt = segmentation_gt[:, z_range, :, :]
            
            image_size = image.shape
            seg_gt_size = segmentation_gt.shape
            assert seg_gt_size[1] == image_size[0] and seg_gt_size[2] == image_size[1] and seg_gt_size[3] == image_size[2]
            
            vertex = np.zeros([samples_per_image, 3, 6]) ## in the order of (1000, 3, z1, z2, y1, y2, x1, x2)
            shapedata = vertex.shape
            
            patch_index = 0 ## update by 1 after generating a patch
            
            while patch_index < samples_per_image:
                # center_z = np.random.randint(0, image_size[0]-int(patch_size[0])) + int(patch_size[0]//2)
                # center_y = np.random.randint(0, image_size[1]-int(patch_size[1])) + int(patch_size[1]//2)
                # center_x = np.random.randint(0, image_size[2]-int(patch_size[2])) + int(patch_size[2]//2)
                
                center_z = np.random.randint(0, image_size[0])
                center_y = np.random.randint(0, image_size[1])
                center_x = np.random.randint(0, image_size[2])
                
                vertex[patch_index][0] = np.array([center_z-int(patch_size[0]//2), center_z+int(patch_size[0]//2),
                                                   center_y-int(patch_size[1]//2), center_y+int(patch_size[1]//2),
                                                   center_x-int(patch_size[2]//2), center_x+int(patch_size[2]//2)])
                
                vertex[patch_index][1] = np.array([center_z-2*int(patch_size[0]//2), center_z+2*int(patch_size[0]//2),
                                                   center_y-2*int(patch_size[1]//2), center_y+2*int(patch_size[1]//2),
                                                   center_x-2*int(patch_size[2]//2), center_x+2*int(patch_size[2]//2)])
                
                vertex[patch_index][2] = np.array([center_z-4*int(patch_size[0]//2), center_z+4*int(patch_size[0]//2),
                                                   center_y-4*int(patch_size[1]//2), center_y+4*int(patch_size[1]//2),
                                                   center_x-4*int(patch_size[2]//2), center_x+4*int(patch_size[2]//2)])
                patch_index += 1
            
            modulo=np.mod(shapedata[0], batch_size)
            if modulo!=0:
                num_to_add=batch_size-modulo
                inds_to_add=np.random.randint(0, shapedata[0], num_to_add) ## the return value is a ndarray
                to_add = vertex[inds_to_add]
                new_vertex = np.vstack((vertex, to_add))
            else:
                new_vertex = vertex.copy()
            
            np.random.shuffle(new_vertex)
            for i_batch in range(int(new_vertex.shape[0]/batch_size)):
                subvertex = new_vertex[i_batch*batch_size:(i_batch+1)*batch_size]
                for count in range(batch_size):
                    ## size_*size_*size_ ##
                    image_one = np.zeros([size_, size_, size_], dtype=np.float32)
                    seg_gt_one = np.zeros([3, size_, size_, size_], dtype=np.float32)
                    seg_gt_one[0] = np.ones([size_, size_, size_], dtype=np.float32)
                    
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
                    
                    ## (4*size_)*(4*size_)*(4*size_) ##
                    image_three = np.zeros([4*size_, 4*size_, 4*size_], dtype=np.float32)

                    copy_from2, copy_to2 = corrected_crop(subvertex[count][2], np.array(list(image_size)))
                    
                    cf_z_lower_bound = int(copy_from2[0])
                    if copy_from2[1] is not None:
                        cf_z_higher_bound = int(copy_from2[1])
                    else:
                        cf_z_higher_bound = None
                                
                    cf_y_lower_bound = int(copy_from2[2])
                    if copy_from2[3] is not None:
                        cf_y_higher_bound = int(copy_from2[3])
                    else:
                        cf_y_higher_bound = None

                    cf_x_lower_bound = int(copy_from2[4])
                    if copy_from2[5] is not None:
                        cf_x_higher_bound = int(copy_from2[5])
                    else:
                        cf_x_higher_bound = None

                    image_three[(copy_to2[0]):(copy_to2[1]),
                                (copy_to2[2]):(copy_to2[3]),
                                (copy_to2[4]):(copy_to2[5])] = \
                                image[cf_z_lower_bound:cf_z_higher_bound,
                                      cf_y_lower_bound:cf_y_higher_bound,
                                      cf_x_lower_bound:cf_x_higher_bound]

                    image_three = np.expand_dims(image_three, axis=0)
                    # seg_gt_one = np.argmax(seg_gt_one, axis=0)
                    seg_gt_one = np.expand_dims(seg_gt_one, axis=0)

                    ## output batch ##
                    image_1 = np.expand_dims(image_one, axis=0)
                    image_2 = np.expand_dims(image_two, axis=0)
                    image_3 = np.expand_dims(image_three, axis=0)

                    if label == 0:
                        Img_1 = image_1
                        Img_2 = image_2
                        Img_3 = image_3
                        seg_gt = seg_gt_one
                        label += 1
                    else:
                        Img_1 = np.vstack((Img_1, image_1))
                        Img_2 = np.vstack((Img_2, image_2))
                        Img_3 = np.vstack((Img_3, image_3))
                        seg_gt = np.vstack((seg_gt, seg_gt_one))
                        label += 1
                    
                    if np.remainder(label, batch_size)==0:
                        mask = np.ones([batch_size, 3, size_, size_, size_], dtype=np.int32)
                        yield ([Img_1, Img_2, Img_3, seg_gt, mask], [])
                        label = 0

def validation_generator(data_path, subject_list, num_classes=3, samples_per_image=100, batch_size=1):
    
    batch_size = batch_size # very important
    
    size_ = 64
    patch_size = np.array([size_, size_, size_]) # very important
    num_of_downpooling = 4 # very important
    patch_stride_regulator = np.array([1, 1, 1]) # this value determines the stride in each dimension when getting the patch; if value = 2, the stride in that dimension is half the value of patch_size
    assert np.all(np.mod(patch_size, 2**num_of_downpooling)) == 0
    stride = patch_size/patch_stride_regulator
    
    data_image_path = os.path.join(data_path + 'images_nii/')
    data_seg_gt_path = os.path.join(data_path + 'label_nii/')

    while True:
        label = 0
        for idx in range(len(subject_list)):
            subject_name = subject_list[idx]


            image = sitk.ReadImage(data_image_path + subject_name)
            image = sitk.GetArrayFromImage(image)
            image = image / np.max(image)

            segmentation_gt = sitk.ReadImage(data_seg_gt_path + subject_name)
            segmentation_gt = sitk.GetArrayFromImage(segmentation_gt)
            segmentation_gt = np.transpose(to_categorical(segmentation_gt, num_classes).reshape(list(segmentation_gt.shape + (num_classes,))), [3, 0, 1, 2])

            image_size = np.array(np.shape(image), dtype=np.int32)

            expanded_image_size = (np.ceil(image_size/(1.0*stride))*stride).astype(np.int32)
            expanded_image = np.zeros(expanded_image_size, dtype=np.float32)
            expanded_image[0:image_size[0], 0:image_size[1], 0:image_size[2]] = image
        
            expanded_seg_gt = np.zeros([3, expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
            expanded_seg_gt[0] = np.ones([expanded_image_size[0], expanded_image_size[1], expanded_image_size[2]], dtype=np.float32)
            expanded_seg_gt[:, 0:image_size[0], 0:image_size[1], 0:image_size[2]] = segmentation_gt
            
            num_of_patch_with_overlapping = (expanded_image_size/stride - patch_stride_regulator + 1).astype(np.int16)
            
            total_num_of_patches = np.prod(num_of_patch_with_overlapping)
            
            num_patch_z = num_of_patch_with_overlapping[0] # used for get patches
            num_patch_y = num_of_patch_with_overlapping[1] # used for get patches
            num_patch_x = num_of_patch_with_overlapping[2] # used for get patches
            
            # print("total number of patches in the image is {0}".format(total_num_of_patches))
        
            center = np.zeros([total_num_of_patches, 3]) ## in the order of (total_num_of_patches, 3) ## (384, 3) ##
            
            patch_index = 0
            
            for ii in range(0, num_patch_z):
                for jj in range(0, num_patch_y):
                    for kk in range(0, num_patch_x):
                        center[patch_index] = np.array([int(ii*stride[0] + patch_size[0] // 2),
                                                        int(jj*stride[1] + patch_size[1] // 2),
                                                        int(kk*stride[2] + patch_size[2] // 2)])
                        patch_index += 1
        
            modulo=np.mod(total_num_of_patches, batch_size)
            
            if modulo!=0:
                num_to_add=batch_size-modulo
                inds_to_add=np.random.randint(0, total_num_of_patches, num_to_add) ## the return value is a ndarray
                to_add = center[inds_to_add]
                new_center = np.vstack((center, to_add))
            else:
                new_center = center.copy()
            
            np.random.shuffle(new_center)
            new_center = new_center[:samples_per_image, :]

            for i_batch in range(int(new_center.shape[0]/batch_size)):
                subvertex = new_center[i_batch*batch_size:(i_batch+1)*batch_size]
                for count in range(batch_size):
                    ## size_*size_*size_ ##
                    image_one = np.zeros([size_, size_, size_], dtype=np.float32)
                    seg_gt_one = np.zeros([3, size_, size_, size_], dtype=np.float32)
                    seg_gt_one[0] = np.ones([size_, size_, size_], dtype=np.float32)
                    
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
                    
                    ## (4*size_)*(4*size_)*(4*size_) ##
                    image_three = np.zeros([4*size_, 4*size_, 4*size_], dtype=np.float32)
                    
                    z_lower_bound = int(subvertex[count][0] - patch_size[0]*2)
                    z_higher_bound = int(subvertex[count][0] + patch_size[0]*2)
                    y_lower_bound = int(subvertex[count][1] - patch_size[1]*2)
                    y_higher_bound = int(subvertex[count][1] + patch_size[1]*2)
                    x_lower_bound = int(subvertex[count][2] - patch_size[2]*2)
                    x_higher_bound = int(subvertex[count][2] + patch_size[2]*2)
    
                    virgin_range = np.array([z_lower_bound, z_higher_bound, y_lower_bound, y_higher_bound, x_lower_bound, x_higher_bound])
                    copy_from2, copy_to2 = corrected_crop(virgin_range, expanded_image_size)
                    
                    cf_z_lower_bound = int(copy_from2[0])
                    if copy_from2[1] is not None:
                        cf_z_higher_bound = int(copy_from2[1])
                    else:
                        cf_z_higher_bound = None
                        
                    cf_y_lower_bound = int(copy_from2[2])
                    if copy_from2[3] is not None:
                        cf_y_higher_bound = int(copy_from2[3])
                    else:
                        cf_y_higher_bound = None
                    
                    cf_x_lower_bound = int(copy_from2[4])
                    if copy_from2[5] is not None:
                        cf_x_higher_bound = int(copy_from2[5])
                    else:
                        cf_x_higher_bound = None
                    
                    image_three[int(copy_to2[0]):copy_to2[1],
                                int(copy_to2[2]):copy_to2[3],
                                int(copy_to2[4]):copy_to2[5]] = \
                                expanded_image[cf_z_lower_bound:cf_z_higher_bound,
                                               cf_y_lower_bound:cf_y_higher_bound,
                                               cf_x_lower_bound:cf_x_higher_bound]
    
                    image_three = np.expand_dims(image_three, axis=0)
                    seg_gt_one = np.expand_dims(seg_gt_one, axis=0)
    
                    ## output batch ##
                    image_1 = np.expand_dims(image_one, axis=0)
                    image_2 = np.expand_dims(image_two, axis=0)
                    image_3 = np.expand_dims(image_three, axis=0)
    
                    if label == 0:
                        Img_1 = image_1
                        Img_2 = image_2
                        Img_3 = image_3
                        seg_gt = seg_gt_one
                        label += 1
                    else:
                        Img_1 = np.vstack((Img_1, image_1))
                        Img_2 = np.vstack((Img_2, image_2))
                        Img_3 = np.vstack((Img_3, image_3))
                        seg_gt = np.vstack((seg_gt, seg_gt_one))
                        label += 1
                    
                    if np.remainder(label, batch_size)==0:
                        mask = np.ones([batch_size, 3, size_, size_, size_], dtype=np.int32)
                        yield ([Img_1, Img_2, Img_3, seg_gt, mask], [])
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
