"""Helper scripts to down-sample Stanford 2D3DS dataset.
"""
import os
import argparse

import PIL.Image as Image
import numpy as np
import cv2


def parse_args():
  """Parsse Command Line Arguments.
  """
  parser = argparse.ArgumentParser(
      description='Helper scripts to down-sample Stanford 2D3DS')
  parser.add_argument('--data_dir', type=str,
                      help='/path/to/Stanford/2D3DS/dir.')
  parser.add_argument('--new_dir', type=str,
                      help='/path/to/down-sampled/Stanford/2D3DS/dir.')

  return parser.parse_args()


def main():
  """Down-sample RGB and Surface Normal.
  """
  args = parse_args()

  dir_names = ['area_1', 'area_2', 'area_3', 'area_4',
               'area_5a', 'area_5b', 'area_6']

  for root_dir_name in dir_names:
    for sub_dir_name in ['rgb', 'normal']:
      dir_name = os.path.join(args.data_dir,
                              root_dir_name,
                              'data',
                              sub_dir_name)
      for dirpath, dirnames, filenames in os.walk(dir_name):
        for file_name in filenames:
          if '.png' not in file_name and '.jpg' not in file_name:
            continue
          arr = np.array(Image.open(os.path.join(dirpath, file_name)))
          h, w = arr.shape[:2]
          new_h, new_w = int(h/2), int(w/2)

          if 'rgb' == sub_dir_name:
            arr = cv2.resize(arr,
                             (new_w,new_h),
                             interpolation=cv2.INTER_LINEAR)
          else:
            arr = cv2.resize(arr,
                             (new_w,new_h),
                             interpolation=cv2.INTER_NEAREST)

          new_dir = dirpath.replace(args.data_dir, args.new_dir)
          if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
          new_name = os.path.join(new_dir, file_name)
          Image.fromarray(arr, mode='RGB').save(new_name)

if __name__ == '__main__':
  main()
