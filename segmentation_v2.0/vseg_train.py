import argparse
import os

from md_segmentation3d.vseg_train import train


def main():

  os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5'
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', nargs='?',
                      default='/home/qinliu19/projects/dental_image_analysis/segmentation_v2.0/config/config.py',
                      help='volumetric segmentation3d train config file')
  args = parser.parse_args()
  train(args.input)


if __name__ == '__main__':
  main()
