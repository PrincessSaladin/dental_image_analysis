import argparse
import md.image3d.python.image3d_io as cio
import md.image3d.python.image3d_tools as ctools
import numpy as np
import os
import pandas as pd

from md.image3d.python.frame3d import Frame3d

def ParseAndCheckArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_folder', type=str,
                      default = '/mnt/disk/MI_project/MI_Aorta/AI031_031_094135/2.0 x 2.0 BodyCT_201',
                      help='The folder of the input dicom.')

  parser.add_argument('--out_spacing', type=float, default=4.0,
                      help='The output image resolution.')

  parser.add_argument('--output_folder', type=str,
                      default='/mnt/disk/Data/unit_test_data/eval_landmark',
                      help='The output folder to save the upright image.')

  parser.add_argument('--input_point_list', type=str,
                      default='/mnt/disk/Data/unit_test_data/eval_landmark/AI031_031_094135_voxel.csv')

  parser.add_argument('--output_point_list',type=str,
                      default='/mnt/disk/Data/unit_test_data/eval_landmark/AI031_031_094135_world.csv')

  parser.add_argument('--output_image', type=str,
                      default='AI031_031_094135.mhd',
                      help='The output image name.')

  args = parser.parse_args()

  if not args.input_folder or not os.path.isdir(args.input_folder):
    raise AssertionError('Please specify the correct input dicom folder: {0}'.format(
      args.input_folder))

  if not args.output_folder:
    raise AssertionError('Please specify the output folder')
  elif not os.path.isdir(args.output_folder):
    os.makefiles(args.output_folder)

  if not args.output_image:
    raise AssertionError('Please specify the output image name.')

  return args

if __name__ == '__main__':
  args = ParseAndCheckArguments()
  image, _ = cio.read_dicom_series(args.input_folder)
  min_world, max_world = image.world_box()

  if args.out_spacing == 0:
    out_spacing = image.spacing()
  else:
    out_spacing = np.ones(3, dtype=np.double) * args.out_spacing

  frame = Frame3d()
  frame.set_origin(min_world)
  frame.set_spacing(out_spacing)

  out_size = np.round((max_world - min_world) / out_spacing)
  out_size = out_size.astype(np.int32)

  out_image = ctools.resample_nn(image, frame, out_size)

  save_path = os.path.join(args.output_folder, args.output_image)
  cio.write_image(out_image, save_path)

  if os.path.isfile(args.input_point_list):
    df = pd.read_csv(args.input_point_list)
    for i in range(4):
      vx = df['x{}'.format(i+1)].tolist()[0]
      vy = df['y{}'.format(i+1)].tolist()[0]
      vz = df['z{}'.format(i+1)].tolist()[0]

      world_coord = image.voxel_to_world([vx,vy,vz])
      print(world_coord)
