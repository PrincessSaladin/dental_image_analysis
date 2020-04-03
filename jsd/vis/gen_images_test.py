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
from jsd.vis.gen_html_report import GenHtmlReport


def parse_and_check_arguments():
  """
  Parse input arguments and raise error if invalid.
  """
  default_image_folder = '/home/qinliu/projects/CT_Dental/data'
  default_label_folder = '/home/qinliu/projects/CT_Dental/landmark'
  default_detection_folder = ''
  default_resolution = [2.0, 2.0, 2.0]
  default_contrast_range = None
  default_output_folder = '/home/qinliu/projects/CT_Dental/landmark_check'

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
  args = parser.parse_args()

  if not os.path.isdir(args.detection_folder):
    print("The detection folder does not exist, so we only check labelled landmarks.")
    usage_flag = 1
  else:
    print("The detection_folder exists, so we compare the labelled and detected landmarks .")
    usage_flag = 2

  return args, usage_flag


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
  
  args, usage_flag = parse_and_check_arguments()

  print("The label folder: {}".format(args.label_folder))
  label_landmark_csvs = glob.glob(os.path.join(args.label_folder, "case_*.csv"))
  label_landmark_csvs.sort()
  print("# landmark files in the label folder: {}\n".format(len(label_landmark_csvs)))
  
  label_landmarks = OrderedDict()
  for label_landmark_csv in label_landmark_csvs:
    file_name = os.path.join(os.path.basename(label_landmark_csv).split('.')[0], 'org.mha')
    landmarks = load_coordinates_from_csv(label_landmark_csv)
    label_landmarks.update({file_name: landmarks})

  if usage_flag == 2:
    print("The detection folder: {}".format(args.detection_folder))
    detection_landmark_csvs = glob.glob(os.path.join(args.detection_folder, "case_*.csv"))
    detection_landmark_csvs.sort()
    print("# landmark files in the detection folder: {}\n".format(len(detection_landmark_csvs)))
    assert len(label_landmark_csvs) == len(detection_landmark_csvs)
    
  gen_plane_images(args.image_folder, label_landmarks, usage_flag, args.contrast_range, args.resolution, args.output_folder)
 
  # landmark_csvs = os.listdir(args.labelled_folder)
  #
  # print("There {0} landmarks to process:".format(len(landmark_csvs)))
  # for landmark_csv in landmark_csvs:
  #   print(landmark_csv.split('.')[0])
  #
  # df = pd.read_csv(args.image_list_csv)
  # num_data = len(df)
  # print("\n{0} data found.\n".format(num_data))
  # file_name_list = df['filename'].tolist()
  #
  #
  # options = Options(resolution=args.resolution,
  #                   contrast_min=args.contrast_min,
  #                   contrast_max=args.contrast_max)
  #
  # for landmark_idx, landmark_csv in enumerate(landmark_csvs):
  #   print("\033[1;32mProcessing landmark {0}... \033[0m".format(landmark_idx + 1))
  #
  #   landmark = landmark_csv.split('.')[0]
  #   output_folder = os.path.join(args.output_folder, landmark)
  #
  #   additional_info = edict()
  #   additional_info.no_lm = []
  #   additional_info.fp1 = []
  #   additional_info.fn1 = []
  #   additional_info.fp2 = []
  #   additional_info.fn2 = []
  #
  #   print("Generating 2D plane images ...")
  #   point_dicts = []
  #   labelled_list_csv = os.path.join(args.labelled_folder, landmark_csv)
  #   labelled_point_dict = LoadCoordinatesFromCSV(labelled_list_csv)
  #   labelled_image_list, additional_info = GeneratePlaneImages(file_name_list, args.input_folder,
  #                                                              labelled_point_dict,
  #                                                              os.path.join(output_folder, "images"),
  #                                                              options, 'labelled', additional_info)
  #   point_dicts.append(labelled_point_dict)
  #
  #
  #   # if the tool is used for error analysis or benchmark.
  #   if usage_flag >= 2:
  #     detected_list_csv = os.path.join(args.detected_folder, landmark_csv)
  #     detected_point_dict = LoadCoordinatesFromCSV(detected_list_csv)
  #     image_type_name = 'detected'
  #     if usage_flag == 3:
  #       image_type_name = 'baseline'
  #     detected_image_list, additional_info = GeneratePlaneImages(file_name_list, args.input_folder,
  #                                                                detected_point_dict,
  #                                                                os.path.join(output_folder, "images"),
  #                                                                options, image_type_name, additional_info)
  #     point_dicts.append(detected_point_dict)
  #
  #
  #   # if the tool is used for benchmark comparison with baseline model.
  #   if usage_flag == 3:
  #     experiment_list_csv = os.path.join(args.experiment_folder, landmark_csv)
  #     experiment_point_dict = LoadCoordinatesFromCSV(experiment_list_csv)
  #     experiment_image_list, additional_info = GeneratePlaneImages(file_name_list, args.input_folder,
  #                                                                  experiment_point_dict,
  #                                                                  os.path.join(output_folder, "images"),
  #                                                                  options, 'experiment', additional_info)
  #     point_dicts.append(experiment_point_dict)
  #
  #
  #   print("Generating html report ...")
  #   GenHtmlReport(file_name_list, point_dicts, additional_info, usage_flag,
  #                 output_folder=output_folder)
