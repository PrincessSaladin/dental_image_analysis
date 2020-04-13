import argparse
import glob
import os

from jsd.vis.gen_images import gen_plane_images, load_coordinates_from_csv
from jsd.vis.gen_html_report import gen_html_report


def parse_and_check_arguments():
  """
  Parse input arguments and raise error if invalid.
  """
  default_image_folder = '/home/qinliu/projects/CT_Dental/data'
  default_label_folder = '/home/qinliu/projects/CT_Dental/landmark'
  default_detection_folder = '/home/qinliu/projects/CT_Dental/results/model_0411_2020'
  default_resolution = [1.5, 1.5, 1.5]
  default_contrast_range = None
  default_output_folder = '/tmp/data/CT_Dental/landmark_html'
  default_generate_pictures = False
  
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
  parser.add_argument('--generate_pictures', type=bool,
                      default=default_generate_pictures,
                      help='Folder containing the generated snapshot images.')
  
  return parser.parse_args()


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
  assert len(label_landmark_csvs) > 0
  label_landmark_csvs.sort()
  print("# landmark files in the label folder: {}".format(len(label_landmark_csvs)))
  
  if usage_flag == 2:
    print("The detection folder: {}".format(args.detection_folder))
    detection_landmark_csvs = glob.glob(os.path.join(args.detection_folder, "case_*.csv"))
    assert len(detection_landmark_csvs) > 0
    detection_landmark_csvs.sort()
    print("# landmark files in the detection folder: {}".format(
      len(detection_landmark_csvs)))
  
    # find the intersection of the labelled and the detected files
    label_landmark_csvs_folder = os.path.dirname(label_landmark_csvs[0])
    detection_landmark_csvs_folder = os.path.dirname(detection_landmark_csvs[0])

    label_landmark_csvs_basenames = []
    for label_landmark_csv in label_landmark_csvs:
      basename = os.path.basename(label_landmark_csv)
      label_landmark_csvs_basenames.append(basename)

    detection_landmark_csvs_basenames = []
    for detection_landmark_csv in detection_landmark_csvs:
      basename = os.path.basename(detection_landmark_csv)
      detection_landmark_csvs_basenames.append(basename)
    
    intersect_basename = \
      list(set(label_landmark_csvs_basenames) & set(detection_landmark_csvs_basenames))
    assert len(intersect_basename) > 0

    label_landmark_csvs, detection_landmark_csvs = [], []
    for basename in intersect_basename:
      label_landmark_csvs.append(os.path.join(label_landmark_csvs_folder, basename))
      detection_landmark_csvs.append(os.path.join(detection_landmark_csvs_folder, basename))
    
    print("# landmark files in the both folder: {}".format(
      len(detection_landmark_csvs)))
    
  label_landmarks = {}
  for label_landmark_csv in label_landmark_csvs:
    file_name = os.path.basename(label_landmark_csv).split('.')[0]
    landmarks = load_coordinates_from_csv(label_landmark_csv)
    label_landmarks.update({file_name: landmarks})
  
  if not os.path.isdir(args.output_folder):
    os.makedirs(args.output_folder)
  
  if usage_flag == 2:
    detection_landmarks = {}
    for detection_landmark_csv in detection_landmark_csvs:
      file_name = os.path.basename(detection_landmark_csv).split('.')[0]
      landmarks = load_coordinates_from_csv(detection_landmark_csv)
      detection_landmarks.update({file_name: landmarks})

  # Generate landmark html report for each landmark.
  landmark_list = [label_landmarks]
  if usage_flag == 2:
    landmark_list.append(detection_landmarks)

  gen_html_report(landmark_list, usage_flag, args.output_folder)
  
  if args.generate_pictures:
    gen_plane_images(args.image_folder, label_landmarks, 'label',
                     args.contrast_range, args.resolution, args.output_folder)
  
    if usage_flag == 2:
      gen_plane_images(args.image_folder, detection_landmarks, 'detection',
                       args.contrast_range, args.resolution, args.output_folder)