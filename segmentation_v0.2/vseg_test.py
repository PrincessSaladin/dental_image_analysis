import argparse

from md_segmentation3d.vseg_apply import segmentation


def main():
  from argparse import RawTextHelpFormatter
  
  
  long_description = 'Segmentation3d Batch Testing Engine\n\n' \
                     'It supports multiple kinds of input:\n' \
                     '1. Image list txt file\n' \
                     '2. Single image file\n' \
                     '3. A folder that contains all testing images\n'
  
  parser = argparse.ArgumentParser(description=long_description,
                                   formatter_class=RawTextHelpFormatter)
  
  parser.add_argument('-i', '--input', type=str,
                      help='input folder/file for intensity images',
                      default='/shenlab/lab_stor6/qinliu/CT_Dental/datasets/test.txt')
  parser.add_argument('-m', '--model', type=str,
                      help='model root folder',
                      default='/shenlab/lab_stor6/qinliu/CT_Dental/models/model_1112_2019')
  parser.add_argument('-o', '--output', type=str,
                      help='output folder for segmentation_v0.1',
                      default='/shenlab/lab_stor6/qinliu/CT_Dental/results/model_1112_2019_epoch3700')
  parser.add_argument('-n', '--seg_name', default='seg.mha',
                      help='the name of the segmentation_v0.1 result to be saved')
  parser.add_argument('-g', '--gpu_id', default='0',
                      help='the gpu id to run model')
  parser.add_argument('--save_image', default=True,
                      help='whether to save original image',
                      action="store_true")
  parser.add_argument('--save_single_prob', default=False,
                      help='whether to save single prob map',
                      action="store_true")
  args = parser.parse_args()
  
  segmentation(args.input, args.model, args.output, args.seg_name,
               int(args.gpu_id), args.save_image,
               args.save_single_prob)


if __name__ == '__main__':
  main()
