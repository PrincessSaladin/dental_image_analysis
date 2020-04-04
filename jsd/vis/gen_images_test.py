import argparse
import numpy as np
import os
import pandas as pd
import sys
from collections import OrderedDict
from easydict import EasyDict as edict
import glob

from jsd.vis.gen_images import GenImagesOptions as Options
from jsd.vis.gen_images import gen_plane_images
from jsd.vis.gen_html_report import gen_html_report


def parse_and_check_arguments():
  """
  Parse input arguments and raise error if invalid.
  """
  default_image_folder = '/home/qinliu/projects/CT_Dental/data'
  default_label_folder = '/home/qinliu/projects/CT_Dental/landmark_debug'
  default_detection_folder = ''
  default_resolution = [1.5, 1.5, 1.5]
  default_contrast_range = None
  default_output_folder = '/tmp/data/CT_Dental/landmark_html'

  parser = argparse.ArgumentParser(
    description='Snapshot three planes centered around landmarks.')
  parser.add_argument('--image_folder', type=str,
                      default=default_image_folder,
                      help='Folder containing the source data.')
  parser.add_argument('--label_folder', type=str,
                      default=default_label_folder,
                      help='A folder where CSV files containing labelled landmark coordinates are stored.')
  parser.add_argument('--detection_folder', type=str,
                      default=default_detection_folder,
                      help='A folder where CSV files containing detected or baseline landmark coordinates are stored.')
  parser.add_argument('--resolution', type=list,
                      default=default_resolution,
                      help="Resolution of the snap shot images.")
  parser.add_argument('--contrast_range', type=list,
                      default=default_contrast_range,
                      help='Minimal and maximal value of contrast intensity window.')
  parser.add_argument('--output_folder', type=str,
                      default=default_output_folder,
                      help='Folder containing the generated snapshot images.')

  return parser.parse_args()


def load_coordinates_from_csv(csv_file):
  """
  Load coordinates (x,y,z) from CSV file and pack them into an OrderedDict.
  Input arguments:
    csv_file: CSV file path containing [x,y,z] coordinates in each row.
  Return:
    Two dimensional array, {idx_0:[x_0, y_0, z_0], ... ,idx_n:[x_n, y_n, z_n]},
    where idx_n is the landmark name.
  """
  assert csv_file.endswith('.csv')

  landmarks = OrderedDict()
  df = pd.read_csv(csv_file)
  for idx in range(len(df)):
    landmark = df.loc[idx]
    landmarks.update({idx:[landmark[0], landmark[1], landmark[2]]})
  
  return landmarks


if __name__ == '__main__':
  
  args = parse_and_check_arguments()
  if not os.path.isdir(args.detection_folder):
    print("The detection folder does not exist, so we only check labelled landmarks.")
    usage_flag = 1
  else:
    print("The detection_folder exists, so we compare the labelled and detected landmarks .")
    usage_flag = 2

  print("The label folder: {}".format(args.label_folder))
  label_landmark_csvs = glob.glob(os.path.join(args.label_folder, "case_*.csv"))
  label_landmark_csvs.sort()
  print("# landmark files in the label folder: {}".format(len(label_landmark_csvs)))
  
  if usage_flag == 2:
    print("The detection folder: {}".format(args.detection_folder))
    detection_landmark_csvs = glob.glob(os.path.join(args.detection_folder, "case_*.csv"))
    detection_landmark_csvs.sort()
    print("# landmark files in the detection folder: {}".format(len(detection_landmark_csvs)))
    assert len(label_landmark_csvs) == len(detection_landmark_csvs)

  label_landmarks = OrderedDict()
  for label_landmark_csv in label_landmark_csvs:
    file_name = os.path.join(
      os.path.basename(label_landmark_csv).split('.')[0], 'org.mha')
    landmarks = load_coordinates_from_csv(label_landmark_csv)
    label_landmarks.update({file_name: landmarks})

  if not os.path.isdir(args.output_folder):
    os.makedirs(args.output_folder)
  gen_plane_images(args.image_folder, label_landmarks, 'label',
                   args.contrast_range, args.resolution, args.output_folder)

  if usage_flag == 2:
    detection_landmarks = OrderedDict()
    for detection_landmark_csv in detection_landmark_csvs:
      file_name = os.path.join(
        os.path.basename(detection_landmark_csv).split('.')[0], 'org.mha')
      landmarks = load_coordinates_from_csv(detection_landmark_csv)
      detection_landmarks.update({file_name: landmarks})
  
    gen_plane_images(args.image_folder, detection_landmarks, 'detection',
                     args.contrast_range, args.resolution, args.output_folder)

  # Generate landmark html report for each landmark.
  landmark_list = [label_landmarks]
  if usage_flag == 2:
    landmark_list.append(detection_landmarks)

  gen_html_report(landmark_list, usage_flag, args.output_folder)