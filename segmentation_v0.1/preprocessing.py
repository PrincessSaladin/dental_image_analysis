#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:18:00 2019

@author: xiaoyangchen
"""

import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as snd

data_folder = '/Users/xiaoyangchen/Desktop/data' # This where images and label are located in

"""

I assume you have two folders under 'data_folder'. One is "images_preprocessing" and the other is "label_preprocessing".
Keep in mind that any image and its corresponding label are assumed to have the same name.
Images and labels are all in nifty format by default.

After processing, I will write processed images and labels into "images" and "label" respectively, saved into nifty format by default.

"""

filenames = list(os.walk(data_folder))

folder = filenames[0][0]

UPPER_BOUND = 3200
LOWER_BOUND = -300

image_file_names = sorted([x for x in list(os.walk(folder + '/images_preprocessing'))[0][2] if x.endswith('.nii.gz')])
label_file_names = sorted([x for x in list(os.walk(folder + '/label_preprocessing'))[0][2] if x.endswith('.nii.gz')])
assert image_file_names == label_file_names, "Inconsistent naming convention"

print("Preprocessing images...")

print("Clipping intensity values -- Intensity normalization -- Spatial normalization...")

common_spacing = np.array([0.4, 0.4, 0.4])

for idx in range(len(image_file_names)):
    print("       Begin processing " + image_file_names[idx])
    
    current = sitk.ReadImage(os.path.join(folder + '/images_preprocessing', image_file_names[idx]))
    spacing = current.GetSpacing()
    
    current_array = (sitk.GetArrayFromImage(current)).astype(np.float32)
    
    resampled = snd.interpolation.zoom(current_array, spacing/common_spacing, order=1)
    clipped = np.clip(resampled, LOWER_BOUND, UPPER_BOUND)
    clipped = (clipped - np.min(clipped))/(np.max(clipped) - np.min(clipped))
        
    clipped_image = sitk.GetImageFromArray(clipped)
    clipped_image.SetSpacing(tuple(common_spacing))
    
    if not os.path.exists(folder + '/images'):
        os.makedirs(folder + '/images/')
    
    sitk.WriteImage(clipped_image, folder + '/images/' + image_file_names[idx])
    
    current_label = sitk.ReadImage(os.path.join(folder + '/label_preprocessing', image_file_names[idx]))
    current_label_array = sitk.GetArrayFromImage(current_label)
    array_shape = current_label_array.shape
    
    unique_values = np.sort(np.unique(current_label_array))
    
    label_splitted = None
    for value_i in range(len(unique_values)):
        label_i = np.zeros_like(current_label_array, dtype=np.float32)
        label_i[np.where(current_label_array == unique_values[value_i])] = 1
        resampled_label_i = (snd.interpolation.zoom(label_i, spacing/common_spacing, order=1) >= 0.5).astype(np.int32)
        resampled_label_i = np.expand_dims(resampled_label_i, axis=0)
        if label_splitted is None:
            label_splitted = resampled_label_i
        else:
            label_splitted = np.concatenate([label_splitted, resampled_label_i], axis=0)

    if not os.path.exists(folder + '/label'):
        os.makedirs(folder + '/label/')
    
    one_map = np.argmax(label_splitted, axis=0)
    one_map = sitk.GetImageFromArray(one_map.astype(np.float32))
    one_map.SetSpacing(tuple(common_spacing))
    sitk.WriteImage(one_map, folder + '/label/' + image_file_names[idx][:-7] + '.nii.gz')
    
    print("       Finished " + image_file_names[idx])
    print("\n")

print("Finished Clipping intensity values -- Intensity normalization -- Spatial normalization.")

print("Finished processing images.")