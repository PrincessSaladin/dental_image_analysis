import numpy as np
import SimpleITK as sitk


def weighted_voxel_center(image, threshold_min, threshold_max):
    """
    Get the weighted voxel center.
    :param image:
    :return:
    """
    assert isinstance(image, sitk.Image)

    image_npy = sitk.GetArrayFromImage(image)
    image_npy[image_npy < threshold_min] = 0
    image_npy[image_npy > threshold_max] = 0
    weight_sum = np.sum(image_npy)
    if weight_sum <= 0:
        return None

    image_npy_x = np.zeros_like(image_npy)
    for i in range(image_npy.shape[0]):
        image_npy_x[i , :, :] = i

    image_npy_y = np.zeros_like(image_npy)
    for j in range(image_npy.shape[1]):
        image_npy_y[:, j, :] = j

    image_npy_z = np.zeros_like(image_npy)
    for k in range(image_npy.shape[2]):
        image_npy_z[:, :, k] = k

    weighted_center_x = np.sum(np.multiply(image_npy, image_npy_x)) / weight_sum
    weighted_center_y = np.sum(np.multiply(image_npy, image_npy_y)) / weight_sum
    weighted_center_z = np.sum(np.multiply(image_npy, image_npy_z)) / weight_sum
    weighted_center = [weighted_center_z, weighted_center_y, weighted_center_x]

    return weighted_center