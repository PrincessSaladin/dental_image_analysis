from __future__ import print_function
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from segmentation3d.dataloader.image_tools import select_random_voxels_in_multi_class_mask, \
  crop_image, convert_image_to_tensor, get_image_frame


def read_landmark_coords(image_name_list, landmark_file_path, target_landmark_label):
  """
  Read a list of labelled landmark csv files and return a list of labelled
  landmarks.
  """
  assert len(image_name_list) == len(landmark_file_path)

  label_dict = {}
  for idx, image_name in enumerate(image_name_list):
    label_dict[image_name] = {}
    label_dict[image_name]['label'] = []
    label_dict[image_name]['name'] = []
    label_dict[image_name]['coords'] = []
    landmark_file_df = pd.read_csv(landmark_file_path[idx])

    for row_idx in range(len(landmark_file_df)):
      landmark_name = landmark_file_df['name'][row_idx]
      if landmark_name in target_landmark_label.keys():
        landmark_label = target_landmark_label[landmark_name]
        x = landmark_file_df['x'][row_idx]
        y = landmark_file_df['y'][row_idx]
        z = landmark_file_df['z'][row_idx]
        landmark_coords = [x, y, z]
        label_dict[image_name]['label'].append(landmark_label)
        label_dict[image_name]['name'].append(landmark_name)
        label_dict[image_name]['coords'].append(landmark_coords)

    assert len(label_dict[image_name]['name']) == len(target_landmark_label.keys())

  return label_dict


def read_image_list(image_list_file, mode):
  """
  Reads the training image list file and returns a list of image file names.
  """
  images_df = pd.read_csv(image_list_file)
  image_name_list = images_df['image_name'].tolist()
  image_path_list = images_df['image_path'].tolist()

  if mode == 'test':
    return image_name_list, image_path_list

  elif mode == 'train' or mode == 'validation':
    landmark_file_path_list = images_df['landmark_file_path'].tolist()
    landmark_mask_path_list = images_df['landmark_mask_path'].tolist()
    organ_mask_path_list = images_df['organ_mask_path'].tolist()
    return image_name_list, image_path_list, landmark_file_path_list, \
           landmark_mask_path_list, organ_mask_path_list

  else:
    raise ValueError('Unsupported mode type.')


class LandmarkDetectionDataset(Dataset):
  """
  Training dataset for multi-landmark detection.
  """
  def __init__(self,
               mode,
               image_list_file,
               target_landmark_label,
               target_organ_label,
               crop_size,
               crop_spacing,
               sampling_method,
               sampling_size,
               positive_upper_bound,
               negative_lower_bound,
               num_pos_patches_per_image,
               num_neg_patches_per_image,
               augmentation_turn_on,
               augmentation_orientation_axis,
               augmentation_orientation_radian,
               augmentation_translation,
               interpolation,
               crop_normalizers):
    self.mode = mode
    assert self.mode == 'train' or self.mode == 'Train'

    self.image_name_list, self.image_path_list, self.landmark_file_path, \
    self.landmark_mask_path, self.organ_mask_path = read_image_list(image_list_file, self.mode)
    assert len(self.image_name_list) == len(self.image_path_list)

    self.target_landmark_label = target_landmark_label
    self.landmark_coords_dict = read_landmark_coords(
      self.image_name_list, self.landmark_file_path, self.target_landmark_label
    )
    self.target_organ_label = target_organ_label
    self.crop_spacing = crop_spacing
    self.crop_size = crop_size
    self.sampling_method = sampling_method
    self.sampling_size = sampling_size
    self.positive_upper_bound = positive_upper_bound
    self.negative_lower_bound = negative_lower_bound
    self.num_pos_patches_per_image = num_pos_patches_per_image
    self.num_neg_patches_per_image = num_neg_patches_per_image
    # + 1 for background
    self.num_landmark_classes = len(target_landmark_label) + 1
    self.num_organ_classes = len(target_organ_label) + 1
    self.augmentation_turn_on = augmentation_turn_on
    self.augmentation_orientation_radian = augmentation_orientation_radian
    self.augmentation_orientation_axis = augmentation_orientation_axis
    self.augmentation_translation = np.array(augmentation_translation, dtype=np.float32)
    self.interpolation = interpolation
    self.crop_normalizers = crop_normalizers

  def __len__(self):
    """ get the number of images in this dataset """
    return len(self.image_name_list)

  def num_modality(self):
    """ get the number of input image modalities """
    return 1

  def num_landmark_classes(self):
    return self.num_landmark_classes

  def num_organ_classes(self):
    return self.num_organ_classes

  def global_sample(self, image):
    """ random sample a position in the image
    :param image: a SimpleITK image object which should be in the RAI coordinate
    :return: a world position in the RAI coordinate
    """
    assert isinstance(image, sitk.Image)

    origin = image.GetOrigin()
    im_size_mm = [image.GetSize()[idx] * image.GetSpacing()[idx] for idx in range(3)]
    crop_size_mm = self.crop_size * self.crop_spacing

    sp = np.array(origin, dtype=np.double)
    for i in range(3):
      if im_size_mm[i] > crop_size_mm[i]:
        sp[i] = origin[i] + np.random.uniform(0, im_size_mm[i] - crop_size_mm[i])
    center = sp + crop_size_mm / 2
    return center

  def center_sample(self, image):
    """ return the world coordinate of the image center
    :param image: a image3d object
    :return: the image center in world coordinate
    """
    assert isinstance(image, sitk.Image)

    origin = image.GetOrigin()
    end_point_voxel = [int(image.GetSize()[idx] - 1) for idx in range(3)]
    end_point_world = image.TransformIndexToPhysicalPoint(end_point_voxel)

    center = np.array([(origin[idx] + end_point_world[idx]) / 2.0 for idx in range(3)], dtype=np.double)
    return center

  def __getitem__(self, index):
    """ get a training sample - image(s) and segmentation pair
    :param index:  the sample index
    :return cropped image, cropped mask, crop frame, case name
    """
    # image IO
    image_name = self.image_name_list[index]
    image_path = self.image_path_list[index]

    images = []
    image = sitk.ReadImage(image_path)
    images.append(image)

    landmark_coords = self.landmark_coords_dict[image_name]

    organ_mask_path = self.organ_mask_path[index]
    organ_mask = sitk.ReadImage(organ_mask_path)

    landmark_mask_path = self.landmark_mask_path[index]
    landmark_mask = sitk.ReadImage(landmark_mask_path)

    # sampling a crop center
    if self.sampling_method == 'CENTER':
      center = self.center_sample(organ_mask)

    elif self.sampling_method == 'GLOBAL':
      center = self.global_sample(organ_mask)

    elif self.sampling_method == 'MASK':
      random_organ_index = np.random.randint(0, self.num_organ_classes - 1)
      centers = select_random_voxels_in_multi_class_mask(
        organ_mask, 1, self.target_organ_label[random_organ_index]
      )
      if len(centers) > 0:
        center = organ_mask.TransformIndexToPhysicalPoint([int(centers[0][idx]) for idx in range(3)])
      else:  # if no segmentation
        center = self.global_sample(organ_mask)

    elif self.sampling_method == 'HYBRID':
      if index % 2:
        center = self.global_sample(organ_mask)
      else:
        random_organ_index = np.random.randint(0, self.num_organ_classes - 1)
        centers = select_random_voxels_in_multi_class_mask(
          organ_mask, 1, self.target_organ_label[random_organ_index]
        )
        if len(centers) > 0:
          center = organ_mask.TransformIndexToPhysicalPoint([int(centers[0][idx]) for idx in range(3)])
        else:  # if no segmentation
          center = self.global_sample(organ_mask)

    else:
      raise ValueError('Only support CENTER, GLOBAL, MASK, and HYBRID sampling methods')

    # random translation
    center += np.random.uniform(-self.augmentation_translation, self.augmentation_translation, size=[3])

    # sample a crop from image and normalize it
    for idx in range(len(images)):
      images[idx] = crop_image(
        images[idx], center, self.crop_size, self.crop_spacing, self.interpolation
      )
      if self.crop_normalizers[idx] is not None:
        images[idx] = self.crop_normalizers[idx](images[idx])

    organ_mask = crop_image(organ_mask, center, self.crop_size, self.crop_spacing, 'NN')
    landmark_mask = crop_image(landmark_mask, center, self.crop_size, self.crop_spacing, 'NN')

    # convert image and masks to tensors
    image_tensor = convert_image_to_tensor(images)
    organ_mask_tensor = convert_image_to_tensor(organ_mask)
    landmark_mask_tensor = convert_image_to_tensor(landmark_mask)

    # convert landmark coords to tensor
    landmark_coords_list = []
    indices = np.argsort(landmark_coords['label'])
    for idx in indices:
      coords = landmark_coords['coords'][idx]
      landmark_coords_list.append([coords[0], coords[1], coords[2]])
    landmark_coords_tensor = torch.from_numpy(np.array(landmark_coords_list, dtype=np.float32))

    # image frame
    image_frame = get_image_frame(images[0])

    return image_tensor, organ_mask_tensor, landmark_mask_tensor, \
           landmark_coords_tensor, image_frame, image_name