import numpy as np


def is_voxel_coordinate_valid(coord_voxel, image_size):
  """
  Check whether the voxel coordinate is out of bound.
  """
  for idx in range(3):
    if coord_voxel[idx] < 0 or coord_voxel[idx] >= image_size[idx]:
      return False
  return True


def is_world_coordinate_valid(coord_world):
  """
  Check whether the world coordinate is valid.
  The world coordinate is invalid if it is (0, 0, 0), (1, 1, 1), or (-1, -1, -1).
  """
  coord_world_npy = np.array(coord_world)
  
  if np.linalg.norm(coord_world_npy, ord=1) < 1e-6 or \
     np.linalg.norm(coord_world_npy - np.ones(3), ord=1) < 1e-6 or \
     np.linalg.norm(coord_world_npy - -1 * np.ones(3), ord=1) < 1e-6:
    return False
  
  return True
