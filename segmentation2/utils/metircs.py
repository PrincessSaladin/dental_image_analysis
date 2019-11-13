import numpy as np


def cal_dsc_binary(gt_npy, seg_npy):
  """
  Calculate dice for binary segmentation
  :param gt_npy:   the numpy of ground truth
  :param seg_npy:  the numpy of segmentation result
  :return:
    Dice: the dice ratio, 1 for perfect segmentation, 0 for missing.
    Type: segmentation type, 'TP' for True Positive, 'TN' for True Negative,
          'FP' for False Positive, 'FN' for False Negative
  """
  isinstance(gt_npy, np.ndarray)
  isinstance(seg_npy, np.ndarray)

  # convert gt and seg to binary mask
  gt_npy[gt_npy < 1] = 0
  gt_npy[gt_npy > 1] = 1
  
  seg_npy[seg_npy < 1] = 0
  seg_npy[seg_npy > 1] = 1

  # determine the type
  min_threshold = 10
  num_voxels_gt, num_voxels_seg = np.sum(gt_npy), np.sum(seg_npy)

  if num_voxels_gt < min_threshold and num_voxels_seg < min_threshold:
    return 1.0, 'TN'
  elif num_voxels_gt < min_threshold and num_voxels_seg >= min_threshold:
    return 0.0, 'FP'
  elif num_voxels_gt >= min_threshold and num_voxels_seg < min_threshold:
    return 0.0, 'FN'
  else:
    intersection = gt_npy + seg_npy
    intersection[intersection == 1] = 0
    num_intersection = np.sum(intersection)
    dsc = num_intersection / (num_voxels_gt + num_voxels_seg)
    return dsc, 'TP'
