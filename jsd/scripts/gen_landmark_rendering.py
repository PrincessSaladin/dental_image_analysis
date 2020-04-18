import ctypes
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk


def int32_to_uint32(i):
    return ctypes.c_uint32(i).value

landmark_folder = '/mnt/projects/CT_Dental/landmark'
segmentation_folder = '/mnt/projects/CT_Dental/data'
image_name = 'case_145_ct_normal'

mask_path = os.path.join(segmentation_folder, image_name, 'seg.mha')
mask = sitk.ReadImage(mask_path)
mask_npy = sitk.GetArrayFromImage(mask)
mask_npy = np.zeros_like(mask_npy, dtype=np.int32)

landmark_file_path = os.path.join(landmark_folder, '{}.csv'.format(image_name))
landmark_df = pd.read_csv(landmark_file_path)
landmark_voxels = []
for index, row in landmark_df.iterrows():
    world = (row['x'], row['y'], row['z'])
    voxel = mask.TransformPhysicalPointToIndex(world)
    landmark_voxels.append(voxel)
    label = index + 2
    mask_npy[voxel[2], voxel[1], voxel[0]] = label
    if voxel[0] >= 1 and voxel[0] < mask.GetSize()[0] - 1 and \
       voxel[1] >= 1 and voxel[1] < mask.GetSize()[1] - 1 and \
       voxel[2] >= 1 and voxel[2] < mask.GetSize()[2] - 1:

        mask_npy[voxel[2] - 1, voxel[1], voxel[0]] = label
        mask_npy[voxel[2] + 1, voxel[1], voxel[0]] = label
        mask_npy[voxel[2], voxel[1] + 1, voxel[0]] = label
        mask_npy[voxel[2], voxel[1] - 1, voxel[0]] = label
        mask_npy[voxel[2], voxel[1], voxel[0] + 1] = label
        mask_npy[voxel[2], voxel[1], voxel[0] - 1] = label

mask_zeros = sitk.GetImageFromArray(mask_npy)
mask_zeros.CopyInformation(mask)
mask_zeros = sitk.Cast(mask_zeros, sitk.sitkInt32)

# find the landmarks in the boundary
landmark_voxels_npy = np.array(landmark_voxels, dtype=np.double)
voxel_x_max = landmark_voxels_npy[np.argmax(landmark_voxels_npy[:, 0])]
voxel_x_min = landmark_voxels_npy[np.argmin(landmark_voxels_npy[:, 0])]
voxel_y_max = landmark_voxels_npy[np.argmax(landmark_voxels_npy[:, 1])]
voxel_y_min = landmark_voxels_npy[np.argmin(landmark_voxels_npy[:, 1])]
voxel_z_max = landmark_voxels_npy[np.argmax(landmark_voxels_npy[:, 2])]
voxel_z_min = landmark_voxels_npy[np.argmin(landmark_voxels_npy[:, 2])]

print(voxel_x_min, voxel_x_max, voxel_y_min, voxel_y_max, voxel_z_min, voxel_z_max)
landmark_mask_folder = '/mnt/projects/CT_Dental/landmark_mask'
landmark_mask_path = os.path.join(landmark_mask_folder, '{}.mha'.format(image_name))
sitk.WriteImage(mask_zeros, landmark_mask_path)
