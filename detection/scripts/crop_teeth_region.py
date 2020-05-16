# This script aims to crop teeth region for teeth landmark detection
# We use five landmarks-landmark 'IC', 'B', 'RMA-R', 'RMA-L', and 'L0'-as
# anchor landmarks for cropping the teeth region.
import os
import pandas as pd
import SimpleITK as sitk

from segmentation3d.utils.image_tools import crop_image
from jsd.utils.landmark_utils import is_voxel_coordinate_valid, is_world_coordinate_valid


def get_file_names(file_folder, prefix_pattern):
    """
    :param file_folder: The input folder.
    :param pattern: The pattern of the target file.
    :return: A list containing file names matched the pattern.
    """
    files = os.listdir(file_folder)
    matched_names = []
    for file in files:
        if file.startswith(prefix_pattern):
            matched_names.append(file)

    return matched_names


image_folder = '/mnt/projects/CT_Dental/data'
cropped_image_save_folder = '/mnt/projects/CT_Dental/data_teeth/'
landmark_csv_folder = '/mnt/projects/CT_Dental/landmark'
landmark_mask_folder = '/mnt/projects/CT_Dental/landmark_mask/batch_4_0.8mm_upper_teeth_batch_2'
anchor_landmarks = ['IC', 'B', 'RMA-R', 'RMA-L', 'L0']
crop_spacing = [0.8, 0.8, 0.8]
crop_size = [128, 96, 96]

# get image name list
csv_file_names = get_file_names(landmark_csv_folder, 'case')
csv_file_names.sort()

for csv_file_name in csv_file_names:
    landmark_df = pd.read_csv(os.path.join(landmark_csv_folder, csv_file_name))

    ic = landmark_df[landmark_df['name'] == 'IC']
    world_coord_ic = [ic['x'].values[0], ic['y'].values[0], ic['z'].values[0]]
    is_world_coord_ic_valid = is_world_coordinate_valid(world_coord_ic)

    b = landmark_df[landmark_df['name'] == 'B']
    world_coord_b = [b['x'].values[0], b['y'].values[0], b['z'].values[0]]
    is_world_coord_b_valid = is_world_coordinate_valid(world_coord_b)

    rmar = landmark_df[landmark_df['name'] == 'RMA-R']
    world_coord_rmar = [rmar['x'].values[0], rmar['y'].values[0], rmar['z'].values[0]]
    is_world_coord_rmar_valid = is_world_coordinate_valid(world_coord_rmar)

    rmal = landmark_df[landmark_df['name'] == 'RMA-L']
    world_coord_rmal = [rmal['x'].values[0], rmal['y'].values[0], rmal['z'].values[0]]
    is_world_coord_rmal_valid = is_world_coordinate_valid(world_coord_rmal)

    l0 = landmark_df[landmark_df['name'] == 'L0']
    world_coord_l0 = [l0['x'].values[0], l0['y'].values[0], l0['z'].values[0]]
    is_world_coord_l0_valid = is_world_coordinate_valid(world_coord_l0)

    image_name = csv_file_name.split('.')[0]
    # if image_name != 'case_25_cbct_patient':
    #     continue

    if is_world_coord_l0_valid:
        image = sitk.ReadImage(os.path.join(image_folder, image_name, 'org.mha'))
        mask = sitk.ReadImage(os.path.join(image_folder, image_name, 'seg.mha'))
        assert isinstance(image, sitk.Image)
        image_size = image.GetSize()
        image_spacing = image.GetSpacing()
        resample_ratio = [crop_spacing[idx] / image_spacing[idx] for idx in range(3)]

        voxel_coord_l0 = image.TransformPhysicalPointToIndex(world_coord_l0)
        print(image_name, world_coord_l0, voxel_coord_l0)


        offset = [-int(64 * resample_ratio[0]), -int(16 * resample_ratio[1]), -int(48 * resample_ratio[2])]
        left_bottom_voxel = [voxel_coord_l0[idx] + offset[idx] for idx in range(3)]
        right_top_voxel = [left_bottom_voxel[idx] + int(crop_size[idx] * resample_ratio[idx]) - 1 for idx in range(3)]
        for idx in range(3):
            left_bottom_voxel[idx] = max(0, left_bottom_voxel[idx])
            right_top_voxel[idx] = min(image_size[idx] - 1, right_top_voxel[idx])

        crop_voxel_center = [(left_bottom_voxel[idx] + right_top_voxel[idx]) // 2 for idx in range(3)]
        crop_world_center = image.TransformContinuousIndexToPhysicalPoint(crop_voxel_center)

        cropped_image = crop_image(image, crop_world_center, [128, 128, 128], crop_spacing, 'LINEAR')
        cropped_mask = crop_image(mask, crop_world_center, [128, 128, 128], crop_spacing, 'NN')
        if not os.path.isdir(os.path.join(cropped_image_save_folder, image_name)):
            os.makedirs(os.path.join(cropped_image_save_folder, image_name))
        # sitk.WriteImage(cropped_image, os.path.join(cropped_image_save_folder, image_name, 'org.mha'))
        sitk.WriteImage(cropped_mask, os.path.join(cropped_image_save_folder, image_name, 'seg.mha'))

        # landmark_mask = sitk.ReadImage(os.path.join(landmark_mask_folder, '{}.mha'.format(image_name)))
        # cropped_landmark_mask = crop_image(landmark_mask, crop_world_center, [128, 128, 128], crop_spacing, 'NN')
        # sitk.WriteImage(cropped_landmark_mask, os.path.join(cropped_image_save_folder, image_name, 'seg_upper_2.mha'))

    else:
        print('Landmark l0 is missing in {}'.format(image_name))