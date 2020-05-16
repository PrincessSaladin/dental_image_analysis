# This script aims to generate datasets for segmentation_v0.2.
import numpy as np
import os
import SimpleITK as sitk
import pandas as pd


def rename_files_for_dataset1():
  data_path = '/home/qinliu/projects/CT_Dental/data_original/CBCT_CT-for-CMF-Segmentation/CBCT-16'
  out_folder = '/home/qinliu/projects/CT_Dental/data_debug'
  
  path_pairs = []
  
  files = os.listdir(data_path)
  files.sort()
  idx = 0
  for file_idx, file in enumerate(files):
    if file.endswith('origin.nii.gz'):
      print('Processing: ', file)
      idx += 1
      image = sitk.ReadImage(os.path.join(data_path, file), sitk.sitkFloat32)
      image = sitk.Cast(image, sitk.sitkFloat32)
      
      mask_label1_path = file.replace('origin', 'midface')
      mask_label1 = sitk.ReadImage(os.path.join(data_path, mask_label1_path))
      
      mask_label2_path = file.replace('origin', 'mandible')
      mask_label2 = sitk.ReadImage(os.path.join(data_path, mask_label2_path))
      
      image_size, image_spacing = np.array(image.GetSize()), np.array(image.GetSpacing())
      mask1_size, mask1_spacing = np.array(mask_label1.GetSize()), np.array(mask_label1.GetSpacing())
      mask2_size, mask2_spacing = np.array(mask_label2.GetSize()), np.array(mask_label2.GetSpacing())
      
      assert np.linalg.norm(image_size - mask1_size) == 0
      assert np.linalg.norm(image_size - mask2_size) == 0
      assert np.linalg.norm(image_spacing - mask1_spacing) < 1e-3
      assert np.linalg.norm(image_spacing - mask2_spacing) < 1e-3
      
      print(image_size, image_spacing, image_size * image_spacing)
      print(image.GetDirection())
      
      # merge label 1 and label2 as a single mask
      mask_label1_npy = sitk.GetArrayFromImage(mask_label1)
      mask_label2_npy = sitk.GetArrayFromImage(mask_label2)
      
      num_voxel_label1 = np.sum(mask_label1_npy)
      num_voxel_label2 = np.sum(mask_label2_npy)
      print('# label1: ', num_voxel_label1)
      print('# label2: ', num_voxel_label2)

      mask_npy = np.zeros_like(mask_label1_npy)
      mask_npy[mask_label1_npy > 0] = 1
      mask_npy[mask_label2_npy > 0] = 2
      mask = sitk.GetImageFromArray(mask_npy)
      mask.CopyInformation(image)
      mask = sitk.Cast(mask, sitk.sitkInt8)
      
      # rename
      image_type = 'cbct'
      patient_type = 'patient'
      # There are four normal subject in this dataset.
      if file.find('02') > 0 or file.find('03') > 0 or file.find('13') > 0 or \
        file.find('14') > 0:
        patient_type = 'normal'
        
      file_name = 'case_{}_{}_{}'.format(idx, image_type, patient_type)
      image_save_folder = os.path.join(out_folder, file_name)
      if not os.path.isdir(image_save_folder):
        os.makedirs(image_save_folder)
      
      sitk.WriteImage(image, os.path.join(image_save_folder, 'org.mha'), True)
      sitk.WriteImage(mask, os.path.join(image_save_folder, 'seg.mha'), True)
      path_pairs.append([os.path.join(data_path, file), image_save_folder])
      
  assert idx == 16
  
  df = pd.DataFrame(data=path_pairs)
  df.to_csv(os.path.join(out_folder, 'path1.csv'))
  
  
def rename_files_for_dataset2():
  data_path = '/home/qinliu/projects/CT_Dental/data_original/CBCT_CT-for-CMF-Segmentation/CBCT-57'
  out_folder = '/home/qinliu/projects/CT_Dental/data_debug'

  path_pairs = []

  files = os.listdir(data_path)
  files.sort()
  idx = 16
  for _, file in enumerate(files):
    if file.endswith('origin.nii.gz'):
      idx += 1
      print(idx, file)
      image = sitk.ReadImage(os.path.join(data_path, file), sitk.sitkFloat32)
      image = sitk.Cast(image, sitk.sitkFloat32)

      mask_label1_path = file.replace('origin', 'midface')
      if os.path.isfile(os.path.join(data_path, mask_label1_path)):
        mask_label1 = sitk.ReadImage(os.path.join(data_path, mask_label1_path))
      else:
        mask_label1 = None

      mask_label2_path = file.replace('origin', 'mandible')
      if os.path.isfile(os.path.join(data_path, mask_label2_path)):
        mask_label2 = sitk.ReadImage(os.path.join(data_path, mask_label2_path))
      else:
        mask_label2 = None

      image_size, image_spacing = np.array(image.GetSize()), np.array(image.GetSpacing())
      if mask_label1 is not None:
        mask1_size, mask1_spacing = np.array(mask_label1.GetSize()), np.array(mask_label1.GetSpacing())
      if mask_label2 is not None:
        mask2_size, mask2_spacing = np.array(mask_label2.GetSize()), np.array(mask_label2.GetSpacing())

      if mask_label1 is not None and mask_label2 is not None:
        assert np.linalg.norm(image_size - mask1_size) == 0
        assert np.linalg.norm(image_size - mask2_size) == 0
        assert np.linalg.norm(image_spacing - mask1_spacing) < 1e-3
        assert np.linalg.norm(image_spacing - mask2_spacing) < 1e-3

      print(image_size, image_spacing, image_size * image_spacing)
      print(image.GetDirection())

      if mask_label1 is not None:
        mask_label1_npy = sitk.GetArrayFromImage(mask_label1)
        num_voxel_label1 = np.sum(mask_label1_npy)
        print('# label1: ', num_voxel_label1)
        mask_npy = np.zeros_like(mask_label1_npy)

      if mask_label2 is not None:
        mask_label2_npy = sitk.GetArrayFromImage(mask_label2)
        num_voxel_label2 = np.sum(mask_label2_npy)
        print('# label2: ', num_voxel_label2)
        mask_npy = np.zeros_like(mask_label2_npy)

      if mask_label1 is not None:
        mask_npy[mask_label1_npy > 0] = 1

      if mask_label2 is not None:
        mask_npy[mask_label2_npy > 0] = 2
      
      mask = sitk.GetImageFromArray(mask_npy)
      mask.CopyInformation(image)
      mask = sitk.Cast(mask, sitk.sitkInt8)
    
      # rename
      image_type = 'cbct'
      patient_type = 'patient'

      file_name = 'case_{}_{}_{}'.format(idx, image_type, patient_type)
      image_save_folder = os.path.join(out_folder, file_name)
      if not os.path.isdir(image_save_folder):
        os.makedirs(image_save_folder)

      sitk.WriteImage(image, os.path.join(image_save_folder, 'org.mha'), True)
      sitk.WriteImage(mask, os.path.join(image_save_folder, 'seg.mha'), True)
      path_pairs.append([os.path.join(data_path, file), image_save_folder])

  assert idx == 90

  df = pd.DataFrame(data=path_pairs)
  df.to_csv(os.path.join(out_folder, 'path2.csv'))


def rename_files_for_dataset2_fix():
  data_path = '/home/qinliu/projects/CT_Dental/data_original/CBCT_CT-for-CMF-Segmentation/CBCT-57'
  out_folder = '/home/qinliu/projects/CT_Dental/data_debug'

  path_pairs = []
  
  files = os.listdir(data_path)
  files.sort()
  for _, file in enumerate(files):
  
    matched = False
    if file.endswith('origina.nii.gz'):
      idx = 91
      image_type = 'cbct'
      patient_type = 'patient'
      pattern = 'origina'
      matched = True
    
    if file.endswith('orgin.nii.gz'):
      idx = 92
      image_type = 'cbct'
      patient_type = 'patient'
      pattern = 'orgin'
      matched = True

    if matched:
      print(file)
      image = sitk.ReadImage(os.path.join(data_path, file), sitk.sitkFloat32)
      image = sitk.Cast(image, sitk.sitkFloat32)
      
      mask_label1_path = file.replace(pattern, 'midface')
      if os.path.isfile(os.path.join(data_path, mask_label1_path)):
        mask_label1 = sitk.ReadImage(os.path.join(data_path, mask_label1_path))
      else:
        mask_label1 = None
      
      mask_label2_path = file.replace(pattern, 'mandible')
      if os.path.isfile(os.path.join(data_path, mask_label2_path)):
        mask_label2 = sitk.ReadImage(os.path.join(data_path, mask_label2_path))
      else:
        mask_label2 = None
        
      print(image.GetSize(), image.GetSpacing())
      
      if mask_label1 is not None:
        mask_label1_npy = sitk.GetArrayFromImage(mask_label1)
        num_voxel_label1 = np.sum(mask_label1_npy)
        print('# label1: ', num_voxel_label1)
        mask_npy = np.zeros_like(mask_label1_npy)
      
      if mask_label2 is not None:
        mask_label2_npy = sitk.GetArrayFromImage(mask_label2)
        num_voxel_label2 = np.sum(mask_label2_npy)
        print('# label2: ', num_voxel_label2)
        mask_npy = np.zeros_like(mask_label2_npy)
      
      if mask_label1 is not None:
        mask_npy[mask_label1_npy > 0] = 1
      
      if mask_label2 is not None:
        mask_npy[mask_label2_npy > 0] = 2
  
      mask = sitk.GetImageFromArray(mask_npy)
      mask.CopyInformation(image)
      mask = sitk.Cast(mask, sitk.sitkInt8)
      
      file_name = 'case_{}_{}_{}'.format(idx, image_type, patient_type)
      image_save_folder = os.path.join(out_folder, file_name)
      if not os.path.isdir(image_save_folder):
        os.makedirs(image_save_folder)

      sitk.WriteImage(image, os.path.join(image_save_folder, 'org.mha'), True)
      sitk.WriteImage(mask, os.path.join(image_save_folder, 'seg.mha'), True)
      path_pairs.append([os.path.join(data_path, file), image_save_folder])

  df = pd.DataFrame(data=path_pairs)
  df.to_csv(os.path.join(out_folder, 'path2_fix.csv'))


def rename_files_for_dataset3():
  data_path = '/home/qinliu/projects/CT_Dental/data_original/CBCT_CT-for-CMF-Segmentation/CBCT-57/CT'
  out_folder = '/home/qinliu/projects/CT_Dental/data_debug'

  path_pairs = []
  
  files = os.listdir(data_path)
  files.sort()
  idx = 92
  for _, file in enumerate(files):
    if file.endswith('origin.nii.gz'):
      idx += 1
      image = sitk.ReadImage(os.path.join(data_path, file), sitk.sitkFloat32)
      image = sitk.Cast(image, sitk.sitkFloat32)
      
      mask_label1_path = file.replace('origin', 'midface')
      mask_label1 = sitk.ReadImage(os.path.join(data_path, mask_label1_path))
      
      mask_label2_path = file.replace('origin', 'mandible')
      mask_label2 = sitk.ReadImage(os.path.join(data_path, mask_label2_path))
      
      print(image.GetSize(), image.GetSpacing())
      
      # merge label 1 and label2 as a single mask
      mask_label1_npy = sitk.GetArrayFromImage(mask_label1)
      mask_label2_npy = sitk.GetArrayFromImage(mask_label2)
      
      num_voxel_label1 = np.sum(mask_label1_npy)
      num_voxel_label2 = np.sum(mask_label2_npy)
      print('# label1: ', num_voxel_label1)
      print('# label2: ', num_voxel_label2)

      mask_npy = np.zeros_like(mask_label1_npy)
      mask_npy[mask_label1_npy > 0] = 1
      mask_npy[mask_label2_npy > 0] = 2
      mask = sitk.GetImageFromArray(mask_npy)
      mask.CopyInformation(image)
      mask = sitk.Cast(mask, sitk.sitkInt8)
      
      # rename
      image_type = 'ct'
      patient_type = 'patient'
      
      file_name = 'case_{}_{}_{}'.format(idx, image_type, patient_type)
      image_save_folder = os.path.join(out_folder, file_name)
      if not os.path.isdir(image_save_folder):
        os.makedirs(image_save_folder)
      
      sitk.WriteImage(image, os.path.join(image_save_folder, 'org.mha'), True)
      sitk.WriteImage(mask, os.path.join(image_save_folder, 'seg.mha'), True)
      path_pairs.append([os.path.join(data_path, file), image_save_folder])

  assert idx == 117

  df = pd.DataFrame(data=path_pairs)
  df.to_csv(os.path.join(out_folder, 'path3.csv'))


def rename_files_for_dataset4():
  data_path = '/home/qinliu/projects/CT_Dental/data_original/CBCT_CT-for-CMF-Segmentation/CT-30'
  out_folder = '/home/qinliu/projects/CT_Dental/data_debug'
  
  path_pairs = []
  
  files = os.listdir(data_path)
  files.sort()
  idx = 117
  for _, file in enumerate(files):
    if file.endswith('origin.nii.gz') or file.endswith('original.nii.gz'):
      print(file)
      idx += 1
      image = sitk.ReadImage(os.path.join(data_path, file), sitk.sitkFloat32)
      image = sitk.Cast(image, sitk.sitkFloat32)
      
      pattern = 'original'
      if file.endswith('origin.nii.gz'):
        pattern = 'origin'

      mask_label1_path = file.replace(pattern, 'midface')
      mask_label1 = sitk.ReadImage(os.path.join(data_path, mask_label1_path))
      
      mask_label2_path = file.replace(pattern, 'mandible')
      mask_label2 = sitk.ReadImage(os.path.join(data_path, mask_label2_path))
      
      # merge label 1 and label2 as a single mask
      mask_label1_npy = sitk.GetArrayFromImage(mask_label1)
      mask_label2_npy = sitk.GetArrayFromImage(mask_label2)
      
      num_voxel_label1 = np.sum(mask_label1_npy)
      num_voxel_label2 = np.sum(mask_label2_npy)
      print('# label1: ', num_voxel_label1)
      print('# label2: ', num_voxel_label2)
      
      mask_npy = np.zeros_like(mask_label1_npy)
      mask_npy[mask_label1_npy > 0] = 1
      mask_npy[mask_label2_npy > 0] = 2
      mask = sitk.GetImageFromArray(mask_npy)
      mask.CopyInformation(image)
      mask = sitk.Cast(mask, sitk.sitkInt8)
      
      # rename
      image_type = 'ct'
      patient_type = 'normal'
      
      file_name = 'case_{}_{}_{}'.format(idx, image_type, patient_type)
      image_save_folder = os.path.join(out_folder, file_name)
      if not os.path.isdir(image_save_folder):
        os.makedirs(image_save_folder)
      
      sitk.WriteImage(image, os.path.join(image_save_folder, 'org.mha'), True)
      sitk.WriteImage(mask, os.path.join(image_save_folder, 'seg.mha'), True)
      path_pairs.append([os.path.join(data_path, file), image_save_folder])

  assert idx == 181

  df = pd.DataFrame(data=path_pairs)
  df.to_csv(os.path.join(out_folder, 'path4.csv'))


# convert case 44
def convert_case_44():
  seg_path = '/home/qinliu/projects/CT_Dental/data_debug/case_44_cbct_patient/seg.mha'
  
  seg = sitk.ReadImage(seg_path)
  seg_npy = sitk.GetArrayFromImage(seg)
  seg_npy[seg_npy == 2] = 1
  seg_transformed = sitk.GetImageFromArray(seg_npy)
  seg_transformed.CopyInformation(seg)
  seg_transformed = sitk.Cast(seg_transformed, sitk.sitkInt8)

  sitk.WriteImage(seg_transformed, seg_path, True)


def dataset_split():
  data_folder = '/home/qinliu/projects/CT_Dental/data'
  dropped_cases_idx = [33, 45, 53, 55, 84, 88, 89, 104, 106, 108, 109]
  valid_cases = []
  for file in os.listdir(data_folder):
    if file.find('case') >= 0:
      is_dropped = False
      for idx in dropped_cases_idx:
        if file.find('case_{}'.format(idx)) >= 0:
          is_dropped = True
          break
      
      if not is_dropped:
        valid_cases.append(file)
  
  # dataset split
  valid_cases.sort()
  num_training = int(len(valid_cases)*0.8)

  training_set = valid_cases[:num_training]
  testing_set = valid_cases[num_training:]

  print(training_set)
  print(testing_set)
  
  datasets_folder = '/home/qinliu/projects/CT_Dental/datasets'
  data_server_folder = '/shenlab/lab_stor6/projects/CT_Dental/data'
  if not os.path.isdir(datasets_folder):
    os.makedirs(datasets_folder)
  
  training_set_file = os.path.join(datasets_folder, 'train_server.txt')
  with open(training_set_file, 'w') as fp:
    name_text = ''
    for file in training_set:
      name_text = name_text + os.path.join(data_server_folder, file, 'org.mha') + '\n'
      name_text = name_text + os.path.join(data_server_folder, file, 'seg.mha') + '\n'

    fp.write(str(len(training_set)) + '\n' + name_text)

  training_set_for_testing_file = \
    os.path.join(datasets_folder, 'train_for_testing_server.txt')
  with open(training_set_for_testing_file, 'w') as fp:
    name_text = ''
    for file in training_set:
      file_path = os.path.join(data_server_folder, file, 'org.mha')
      name_text += file + ' ' + file_path + '\n'

    fp.write(str(len(training_set)) + '\n' + name_text)

  testing_set_file = os.path.join(datasets_folder, 'test_server.txt')
  with open(testing_set_file, 'w') as fp:
    name_text = ''
    for file in testing_set:
      file_path = os.path.join(data_server_folder, file, 'org.mha')
      name_text += file + ' ' + file_path + '\n'
      
    fp.write(str(len(testing_set)) + '\n' + name_text)


def data_identity_check():
  data_folder = '/home/qinliu/projects/CT_Dental/data'
  data_debug_folder = '/home/qinliu/projects/CT_Dental/data_debug'

  files = os.listdir(data_folder)
  for file in files:
    basename = os.path.basename(file)
    if basename.find('case') == 0:
      print(basename)
      data_img = sitk.ReadImage(os.path.join(data_folder, basename, 'org.mha'))
      data_img_npy = sitk.GetArrayFromImage(data_img)
      
      data_debug_img = sitk.ReadImage(os.path.join(data_debug_folder, basename, 'org.mha'))
      data_debug_img_npy = sitk.GetArrayFromImage(data_debug_img)
      assert np.max(data_debug_img_npy - data_img_npy) < 1e-6
      
      data_seg = sitk.ReadImage(os.path.join(data_folder, basename, 'seg.mha'))
      data_seg_npy = sitk.GetArrayFromImage(data_seg)
      
      data_debug_seg = sitk.ReadImage(os.path.join(data_debug_folder, basename, 'seg.mha'))
      data_debug_seg_npy = sitk.GetArrayFromImage(data_debug_seg)
      assert np.max(data_debug_seg_npy - data_seg_npy) < 1e-6


def merge_all_path_files_into_a_sigle_file():
  files_folder = '/home/qinliu/projects/CT_Dental/data_debug/'
  file1 = os.path.join(files_folder, 'path1.csv')
  df = pd.read_csv(file1, usecols=[1,2], header=None, skiprows=[0])
  
  file2 = os.path.join(files_folder, 'path2.csv')
  df2 = pd.read_csv(file2, usecols=[1,2], header=None, skiprows=[0])
  df = df.append(df2, sort=False)
  
  file3 = os.path.join(files_folder, 'path2_fix.csv')
  df3 = pd.read_csv(file3, usecols=[1,2], header=None, skiprows=[0])
  df = df.append(df3, sort=False)

  file4 = os.path.join(files_folder, 'path3.csv')
  df4 = pd.read_csv(file4, usecols=[1,2], header=None, skiprows=[0])
  df = df.append(df4, sort=False)
  
  file5 = os.path.join(files_folder, 'path4.csv')
  df5 = pd.read_csv(file5, usecols=[1,2], header=None, skiprows=[0])
  df = df.append(df5, sort=False)
  
  output_file_folder = os.path.join(files_folder, 'path.csv')
  df.to_csv(output_file_folder, index=False, header=['source', 'destination'])


def check_missing_landmark_files():
  data_folder = '/home/qinliu/projects/CT_Dental/data'
  landmark_folder = '/home/qinliu/projects/CT_Dental/landmark'
  
  data_list = os.listdir(data_folder)
  for data in data_list:
    if data.startswith('case'):
      landmark_file = os.path.join(landmark_folder, '{}.csv'.format(data))
      if not os.path.isfile(landmark_file):
        print(data)


if __name__ == "__main__":
  
  steps = [10]
  
  if 1 in steps:
    rename_files_for_dataset1()
    
  if 2 in steps:
    rename_files_for_dataset2()
    
  if 3 in steps:
    rename_files_for_dataset2_fix()

  if 4 in steps:
    rename_files_for_dataset3()

  if 5 in steps:
    rename_files_for_dataset4()

  if 6 in steps:
    convert_case_44()

  if 7 in steps:
    dataset_split()
    
  if 8 in steps:
    data_identity_check()
    
  if 9 in steps:
    merge_all_path_files_into_a_sigle_file()
    
  if 10 in steps:
    check_missing_landmark_files()
    