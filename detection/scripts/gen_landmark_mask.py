from __future__ import print_function
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd

from segmentation3d.dataloader.image_tools import resample_spacing


def gen_single_landmark_mask(ref_image, landmark_df, spacing, pos_upper_bound, neg_lower_bound):
  assert isinstance(ref_image, sitk.Image)

  ref_image = resample_spacing(ref_image, spacing, 1, 'NN')
  ref_image_npy = sitk.GetArrayFromImage(ref_image)
  ref_image_size = ref_image.GetSize()
  landmark_mask_npy = np.zeros_like(ref_image_npy)
  for landmark_name in landmark_df.keys():
    landmark_label = landmark_df[landmark_name]['label']
    landmark_world = [landmark_df[landmark_name]['x'],
                      landmark_df[landmark_name]['y'],
                      landmark_df[landmark_name]['z']]
    landmark_voxel = ref_image.TransformPhysicalPointToIndex(landmark_world)
    for x in range(landmark_voxel[0] - neg_lower_bound,
                   landmark_voxel[0] + neg_lower_bound):
      for y in range(landmark_voxel[1] - neg_lower_bound,
                     landmark_voxel[1] + neg_lower_bound):
        for z in range(landmark_voxel[2] - neg_lower_bound,
                       landmark_voxel[2] + neg_lower_bound):
          if x < 0 or x >= ref_image_size[0] or \
             y < 0 or y >= ref_image_size[1] or \
             z < 0 or z >= ref_image_size[2]:
            continue

          distance = np.linalg.norm(np.array([x, y, z], dtype=np.float32) - landmark_voxel)
          if distance < pos_upper_bound:
            landmark_mask_npy[z, y, x] = float(landmark_label)
          elif distance < neg_lower_bound and abs(landmark_mask_npy[z, y, x]) < 1e-6:
            landmark_mask_npy[z, y, x] = -1.0

  landmark_mask = sitk.GetImageFromArray(landmark_mask_npy)
  landmark_mask.CopyInformation(ref_image)

  return landmark_mask


def gen_landmark_batch_1_2mm():
  target_landmark_label = {
    'S': 1,
    'Gb': 2,
    'Rh': 4,
    'Fz-R': 7,
    'Fz-L': 8,
    'Ba': 21,
    'FMP': 22,
    'Zy-R': 30,
    'Zy-L': 33,
    'U0': 44,
    'Me': 83,
    'L7DBC-R': 135,
    'L7DBC-L': 136
  }

  spacing = [2, 2, 2]  # mm
  pos_upper_bound = 3  # voxel
  neg_lower_bound = 6  # voxel

  # get image name list
  landmark_folder = '/mnt/projects/CT_Dental/landmark'
  landmark_files = os.listdir(landmark_folder)
  image_names = []
  for landmark_file in landmark_files:
    if landmark_file.startswith('case'):
      image_names.append(landmark_file.split('.')[0])

  image_folder = '/mnt/projects/CT_Dental/data'
  image_out_folder = '/mnt/projects/CT_Dental/landmark_mask/batch_1_2mm'
  if not os.path.isdir(image_out_folder):
    os.makedirs(image_out_folder)

  for image_name in image_names:
    print(image_name)
    landmark_df = pd.read_csv(os.path.join(landmark_folder, '{}.csv'.format(image_name)))
    target_landmark_df = {}
    for landmark_name in target_landmark_label.keys():
      target_landmark_df[landmark_name] = {}
      landmark_label = target_landmark_label[landmark_name]
      x = landmark_df[landmark_df['name'] == landmark_name]['x'].values[0]
      y = landmark_df[landmark_df['name'] == landmark_name]['y'].values[0]
      z = landmark_df[landmark_df['name'] == landmark_name]['z'].values[0]
      target_landmark_df[landmark_name]['label'] = landmark_label
      target_landmark_df[landmark_name]['x'] = float(x)
      target_landmark_df[landmark_name]['y'] = float(y)
      target_landmark_df[landmark_name]['z'] = float(z)

    image = sitk.ReadImage(os.path.join(image_folder, image_name, 'org.mha'))
    landmark_mask = gen_single_landmark_mask(
      image, target_landmark_df, spacing, pos_upper_bound, neg_lower_bound
    )

    sitk.WriteImage(landmark_mask, os.path.join(image_out_folder, '{}.mha'.format(image_name)))

if __name__ == '__main__':

  steps = [1]

  if 1 in steps:
    gen_landmark_batch_1_2mm()