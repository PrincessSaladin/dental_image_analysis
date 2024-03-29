import argparse
import glob
import importlib
import torch.nn as nn
import os
import pandas as pd
import SimpleITK as sitk
import time
import torch
from easydict import EasyDict as edict

from segmentation3d.utils.file_io import load_config, readlines
from segmentation3d.utils.model_io import get_checkpoint_folder
from segmentation3d.dataloader.image_tools import convert_image_to_tensor, \
  image_partition_by_fixed_size, resample_spacing, crop_image
from segmentation3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer
from jsd.utils.landmark_utils import is_voxel_coordinate_valid, \
  is_world_coordinate_valid


def read_test_txt(txt_file):
  """ read single-modality txt file
  :param txt_file: image list txt file path
  :return: a list of image path list, list of image case names
  """
  lines = readlines(txt_file)
  case_num = int(lines[0])
  
  if len(lines) - 1 != case_num:
    raise ValueError('case num do not equal path num!')
  
  file_name_list, file_path_list = [], []
  for i in range(case_num):
    im_msg = lines[1 + i]
    im_msg = im_msg.strip().split()
    im_name = im_msg[0]
    im_path = im_msg[1]
    if not os.path.isfile(im_path):
      raise ValueError('image not exist: {}'.format(im_path))
    file_name_list.append(im_name)
    file_path_list.append(im_path)
  
  return file_name_list, file_path_list


def read_test_folder(folder_path):
  """ read single-modality input folder
  :param folder_path: image file folder path
  :return: a list of image path list, list of image case names
  """
  suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
  file = []
  for suf in suffix:
    file += glob.glob(os.path.join(folder_path, '*' + suf))
  
  file_name_list, file_path_list = [], []
  for im_pth in sorted(file):
    _, im_name = os.path.split(im_pth)
    for suf in suffix:
      idx = im_name.find(suf)
      if idx != -1:
        im_name = im_name[:idx]
        break
    file_name_list.append(im_name)
    file_path_list.append(im_pth)
  
  return file_name_list, file_path_list


def load_seg_model(model_folder, gpu_id=0):
  """ load segmentation model from folder
  :param model_folder:    the folder containing the segmentation model
  :param gpu_id:          the gpu device id to run the segmentation model
  :return: a dictionary containing the model and inference parameters
  """
  assert os.path.isdir(model_folder), 'Model folder does not exist: {}'.format(
    model_folder)
  
  # load inference config file
  latest_checkpoint_dir = get_checkpoint_folder(
    os.path.join(model_folder, 'checkpoints'), -1)
  infer_cfg = load_config(
    os.path.join(latest_checkpoint_dir, 'infer_config.py'))
  model = edict()
  model.infer_cfg = infer_cfg
  
  # load model state
  chk_file = os.path.join(latest_checkpoint_dir, 'params.pth')
  
  if gpu_id >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(int(gpu_id))
    # load network module
    state = torch.load(chk_file)
    net_module = importlib.import_module(
      'jsd.network.' + state['net'])
    net = net_module.RegressionNet(state['crop_size'], state['in_channels'], state['num_landmarks'] * 3)
    net = nn.parallel.DataParallel(net, device_ids=[0])
    net.load_state_dict(state['state_dict'])
    net.eval()
    net = net.cuda()
    del os.environ['CUDA_VISIBLE_DEVICES']
  
  else:
    state = torch.load(chk_file, map_location='cpu')
    net_module = importlib.import_module(
      'jsd.network.' + state['net'])
    net = net_module.RegressionNet(state['crop_size'], state['in_channels'], state['num_landmarks'] * 3)
    net.load_state_dict(state['state_dict'])
    net.eval()
  
  model.net = net
  model.crop_size, model.crop_spacing, model.max_stride, model.interpolation = \
    state['crop_size'], state['crop_spacing'], state['max_stride'], state['interpolation']
  model.in_channels, model.num_classes, model.num_landmarks = \
    state['in_channels'], state['num_classes'], state['num_landmarks']
  
  model.crop_normalizers = []
  for crop_normalizer in state['crop_normalizers']:
    if crop_normalizer['type'] == 0:
      mean, stddev, clip = crop_normalizer['mean'], crop_normalizer['stddev'], \
                           crop_normalizer['clip']
      model.crop_normalizers.append(FixedNormalizer(mean, stddev, clip))
    
    elif crop_normalizer['type'] == 1:
      clip_sigma = crop_normalizer['clip_sigma']
      model.crop_normalizers.append(AdaptiveNormalizer(clip_sigma))
    
    else:
      raise ValueError('Unsupported normalization type.')
  
  return model


def segmentation_voi(model, iso_image, start_voxel, end_voxel, use_gpu):
  """ Segment the volume of interest
  :param model:           the loaded segmentation model.
  :param iso_image:       the image volume that has the same spacing with the model's resampling spacing.
  :param start_voxel:     the start voxel of the volume of interest (inclusive).
  :param end_voxel:       the end voxel of the volume of interest (exclusive).
  :param use_gpu:         whether to use gpu or not, bool type.
  :return:
    mean_prob_maps:        the mean probability maps of all classes
    std_maps:              the standard deviation maps of all classes
  """
  assert isinstance(iso_image, sitk.Image)
  
  roi_image = iso_image[start_voxel[0]:end_voxel[0],
              start_voxel[1]:end_voxel[1], start_voxel[2]:end_voxel[2]]
  
  if model['crop_normalizers'] is not None:
    roi_image = model.crop_normalizers[0](roi_image)
  
  roi_image_tensor = convert_image_to_tensor(roi_image).unsqueeze(0)
  if use_gpu:
    roi_image_tensor = roi_image_tensor.cuda()
  
  with torch.no_grad():
    landmarks_pred = model['net'](roi_image_tensor)
    
  return landmarks_pred


def segmentation(input_path, model_folder, output_folder, gpu_id):
  """ volumetric image segmentation engine
  :param input_path:          The path of text file, a single image file or a root dir with all image files
  :param model_folder:        The path of trained model
  :param output_folder:       The path of out folder
  :param gpu_id:              Which gpu to use, by default, 0
  :return: None
  """
  
  # load model
  begin = time.time()
  model = load_seg_model(model_folder, gpu_id)
  load_model_time = time.time() - begin
  
  # load test images
  if os.path.isfile(input_path):
    if input_path.endswith('.txt'):
      file_name_list, file_path_list = read_test_txt(input_path)
    else:
      if input_path.endswith('.mhd') or input_path.endswith(
          '.mha') or input_path.endswith('.nii.gz') or \
          input_path.endswith('.nii') or input_path.endswith(
        '.hdr') or input_path.endswith('.image3d'):
        im_name = os.path.basename(input_path)
        file_name_list = [im_name]
        file_path_list = [input_path]
      
      else:
        raise ValueError('Unsupported input path.')
  
  elif os.path.isdir(input_path):
    file_name_list, file_path_list = read_test_folder(input_path)
  
  else:
    raise ValueError('Unsupported input path.')

  if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
    
  # test each case
  num_success_case = 0
  total_inference_time = 0
  for i, file_path in enumerate(file_path_list):
    print('{}: {}'.format(i, file_path))
    
    # load image
    begin = time.time()
    image = sitk.ReadImage(file_path, sitk.sitkFloat32)
    read_image_time = time.time() - begin
    
    iso_image = resample_spacing(image, model['crop_spacing'], model['max_stride'],
                                 model['interpolation'])
    assert isinstance(iso_image, sitk.Image)
    iso_center = iso_image.TransformIndexToPhysicalPoint([iso_image.GetSize()[idx] // 2 for idx in range(3)])
    iso_image = crop_image(iso_image, iso_center, model['crop_size'],
                           model['crop_spacing'], model['interpolation'])
       
    partition_type = model['infer_cfg'].general.partition_type
    partition_stride = model['infer_cfg'].general.partition_stride
    if partition_type == 'DISABLE':
      start_voxels = [[0, 0, 0]]
      end_voxels = [[int(iso_image.GetSize()[idx]) for idx in range(3)]]
    
    elif partition_type == 'SIZE':
      partition_size = model['infer_cfg'].general.partition_size
      max_stride = model['max_stride']
      start_voxels, end_voxels = \
        image_partition_by_fixed_size(iso_image, partition_size,
                                      partition_stride, max_stride)
    
    else:
      raise ValueError('Unsupported partition type!')
    
    begin = time.time()
    voi_landmarks_preds = []
    for idx in range(len(start_voxels)):
      start_voxel, end_voxel = start_voxels[idx], end_voxels[idx]
      
      voi_landmarks_pred = \
        segmentation_voi(model, iso_image, start_voxel, end_voxel, gpu_id > 0)
      
      voi_landmarks_preds.append(voi_landmarks_pred)
      print('{:0.2f}%'.format((idx + 1) / len(start_voxels) * 100))

    inference_time = time.time() - begin
    
    begin = time.time()
    
    # convert to csv file
    landmarks_pred = voi_landmarks_preds[0].cpu().numpy()
    batch_size, num_landmark_coords = landmarks_pred.shape
    num_landmarks = model['num_landmarks']
    assert batch_size == 1 and num_landmark_coords == num_landmarks * 3
    
    landmarks_content = []
    iso_image_size = iso_image.GetSize()
    for landmark_idx in range(num_landmarks):
      landmark_norm_coord = [landmarks_pred[0][3*landmark_idx + 0],
                             landmarks_pred[0][3*landmark_idx + 1],
                             landmarks_pred[0][3*landmark_idx + 2]]
      landmark_voxel_coord = [landmark_norm_coord[idx] * iso_image_size[idx] for idx in range(3)]
      landmark_world_coord = iso_image.TransformContinuousIndexToPhysicalPoint(landmark_voxel_coord)
      for coord_idx in range(3):
        if not is_voxel_coordinate_valid(landmark_voxel_coord, iso_image_size) or \
            not is_world_coordinate_valid(landmark_world_coord):
          landmark_world_coord = [-1, -1, -1]
      
      landmarks_content.append(landmark_world_coord)
    
    # save results
    df = pd.DataFrame(data=landmarks_content, columns=['x', 'y', 'z'])
    save_name = '{}.csv'.format(file_name_list[i])
    df.to_csv(os.path.join(output_folder, save_name), index=False)
    
    save_time = time.time() - begin
    
    total_test_time = load_model_time + read_image_time + inference_time + save_time
    total_inference_time += inference_time
    num_success_case += 1
    
    print('total test time: {:.2f}, average inference time: {:.2f}'.format(
      total_test_time, total_inference_time / num_success_case))


def main():
  long_description = 'Inference engine for 3d medical image segmentation \n' \
                     'It supports multiple kinds of input:\n' \
                     '1. Single image\n' \
                     '2. A text file containing paths of all testing images\n' \
                     '3. A folder containing all testing images\n'

  default_input = '/shenlab/lab_stor6/qinliu/CT_Dental/datasets/test_server.txt'
  default_model = '/shenlab/lab_stor6/qinliu/CT_Dental/models/model_0411_2020'
  default_output = '/shenlab/lab_stor6/qinliu/CT_Dental/results/model_0411_2020/epoch_2000'
  default_gpu_id = 5
  
  parser = argparse.ArgumentParser(description=long_description)
  parser.add_argument('-i', '--input', default=default_input,
                      help='input folder/file for intensity images')
  parser.add_argument('-m', '--model', default=default_model,
                      help='model root folder')
  parser.add_argument('-o', '--output', default=default_output,
                      help='output folder for segmentation')
  parser.add_argument('-g', '--gpu_id', type=int, default=default_gpu_id,
                      help='the gpu id to run model, set to -1 if using cpu only.')
  
  args = parser.parse_args()
  segmentation(args.input, args.model, args.output, args.gpu_id)


if __name__ == '__main__':
  main()
