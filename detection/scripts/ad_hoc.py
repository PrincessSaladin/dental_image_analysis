from PIL import Image
import glob
import numpy as np
import os
import SimpleITK as sitk

from segmentation3d.utils.image_tools import resample_spacing
from detection.utils.landmark_utils import merge_landmark_files

def resample_single_image():
    image_path = '/mnt/projects/CT_Dental/data/case_174_ct_normal/org.mha'
    image = sitk.ReadImage(image_path)
    resampled_image = resample_spacing(image, [0.4, 0.4, 0.4], 1, 'LINEAR')
    image_save_path = '/mnt/projects/CT_Dental/landmark_mask/batch_4_0.4mm_lower_teeth/case_174_ct_normal_org.mha'
    sitk.WriteImage(resampled_image, image_save_path)


    mask_path = '/mnt/projects/CT_Dental/data/case_174_ct_normal/seg.mha'
    mask = sitk.ReadImage(mask_path)
    resampled_mask = resample_spacing(mask, [0.4, 0.4, 0.4], 1, 'NN')
    mask_save_path = '/mnt/projects/CT_Dental/landmark_mask/batch_4_0.4mm_lower_teeth/case_174_ct_normal_seg.mha'
    sitk.WriteImage(resampled_mask, mask_save_path)


def vis_landmarks():
    landmark_mask_path = '/mnt/projects/CT_Dental/landmark_mask/batch_4_0.4mm_upper_teeth/case_174_ct_normal.mha'
    landmark_mask = sitk.ReadImage(landmark_mask_path)
    landmark_mask_npy = sitk.GetArrayFromImage(landmark_mask)
    landmark_mask_npy[landmark_mask_npy < -0.001] = 2
    landmark_mask_npy[np.where((landmark_mask_npy > 0) & (landmark_mask_npy <= 74))] = 2

    label_path = '/mnt/projects/CT_Dental/landmark_mask/batch_4_0.4mm_upper_teeth/case_174_ct_normal_seg.mha'
    label = sitk.ReadImage(label_path)
    label_npy = sitk.GetArrayFromImage(label)
    label_npy[label_npy > 1] = 0

    merged_label_npy = label_npy + landmark_mask_npy
    merged_label = sitk.GetImageFromArray(merged_label_npy)
    merged_label.CopyInformation(landmark_mask)

    merged_label_save_path = '/mnt/projects/CT_Dental/landmark_mask/batch_4_0.4mm_upper_teeth/case_174_ct_normal_upper_vis.mha'
    sitk.WriteImage(merged_label, merged_label_save_path)


def gen_gif_from_pngs():

    png_folder = '/mnt/projects/CT_Dental/landmark_mask/vis'

    # Create the frames
    frames = []
    imgs = glob.glob(os.path.join(png_folder, 'lower_teeth*.png'))
    imgs.sort()
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(os.path.join(png_folder, 'landmark_lower_teeth.gif'),
                   format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=500, loop=0)


def test_merge_landmark_files():
    landmark_folder = '/mnt/projects/CT_Dental/results/model_0502_2020/epoch_2000/Pre_Post_Facial_Data-Ma'
    landmark_batch_1 = 'batch_1'
    landmark_batch_2 = 'batch_2'
    landmark_batch_3 = 'batch_3'

    merged_landmark_folder = '/mnt/projects/CT_Dental/results/model_0502_2020/epoch_2000/Pre_Post_Facial_Data-Ma/batch_merged'
    landmark_file_names = os.listdir(os.path.join(landmark_folder, landmark_batch_1))
    for landmark_file_name in landmark_file_names:
        print(landmark_file_name)
        landmark_files = []
        landmark_files.append(os.path.join(landmark_folder, landmark_batch_1, landmark_file_name))
        landmark_files.append(os.path.join(landmark_folder, landmark_batch_2, landmark_file_name))
        landmark_files.append(os.path.join(landmark_folder, landmark_batch_3, landmark_file_name))

        landmark_merged_path = os.path.join(merged_landmark_folder, landmark_file_name)
        merge_landmark_files(landmark_files, landmark_merged_path)


if __name__ == '__main__':

    # vis_landmarks()

    #gen_gif_from_pngs()

    # resample_single_image()

    test_merge_landmark_files()