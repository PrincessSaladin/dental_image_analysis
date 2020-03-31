from md_segmentation3d.vis.checkseg import batch_dump_seg_result_folder


def main():
  import argparse
  from argparse import RawTextHelpFormatter
  
  long_description = \
    'Batch visualize segmentation_v1.0 results ' \
    'from all cases in the folder\n' \
    'The folder structure should be like:\n' \
    'Top-folder\n' \
    '  - case 1\n' \
    '    - org.mha (by default)\n' \
    '    - seg.mha (by default)\n' \
    '  - case 2\n' \
    '  - ...\n' \
    '  - case N\n'
  
  parser = argparse.ArgumentParser(description=long_description,
                                   formatter_class=RawTextHelpFormatter)
  
  parser.add_argument('-i', '--input',
                      default=r'/shenlab/lab_stor6/qinliu/CT_Dental/results/model_1112_2019_epoch3700',
                      help='the top-level folder')
  
  parser.add_argument('-c', '--config',
                      default=r'/shenlab/lab_stor6/qinliu/CT_Dental/data_check/label_mandible/segdump.ini',
                      help='the config file for batch visualization')
  
  parser.add_argument('-o', '--output',
                      default=r'/shenlab/lab_stor6/qinliu/CT_Dental/results/model_1112_2019_epoch3700_html_report',
                      help='the output folder')
  
  parser.add_argument('--intensity-name',
                      default='org.mha',
                      help='the full name of output intensity image')
  
  parser.add_argument('--segmentation_v1.0-name',
                      default='seg.mha',
                      help='the full name of output segmentation_v1.0')
  
  parser.add_argument('--method',
                      default='2',
                      help='2 is interval slice,1 is center slice')
  
  parser.add_argument('--directory-level',
                      default=2,
                      type=int,
                      help='the number of directory levels used as case name')
  
  args = parser.parse_args()
  
  batch_dump_seg_result_folder(folder=args.input,
                               config_file=args.config,
                               outputfolder=args.output,
                               orgkey=args.intensity_name,
                               segkey=args.segmentation_name,
                               method=args.method,
                               levels=args.directory_level)


if __name__ == '__main__':
  main()