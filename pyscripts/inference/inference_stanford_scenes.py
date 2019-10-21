"""Inference scripts for Stanford 2D3DS dataset.
"""
from __future__ import print_function

import argparse
import os
import time
import math

import tensorflow as tf
import numpy as np
import scipy.io
from PIL import Image

#from stanford_scenes.models.fcrn import structured_predictor
from stanford_scenes.models.structured_predictor import structured_predictor
from stanford_scenes.image_reader import ImageReader
import utils.general

IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)


def get_arguments():
  """Parse all the arguments provided from the CLI.
    
  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description='Inference Segmentation, Depth, Surface Normals on NYUv2')
  parser.add_argument('--data_dir', type=str, default='',
                      help='/path/to/dataset/.')
  parser.add_argument('--data_list', type=str, default='',
                      help='/path/to/datalist/file.')
  parser.add_argument('--input_size', type=str, default='512,512',
                      help='Comma-separated string with H and W of image.')
  parser.add_argument('--depth_unit', type=float, default=512.0,
                      help='Each pixel value difference means 1/depth_unit '
                           'meters.')
  parser.add_argument('--train_segmentation', action='store_true',
                      help='Whether to predict segmentations.')
  parser.add_argument('--train_depth', action='store_true',
                      help='Whether to predict depths.')
  parser.add_argument('--train_normal', action='store_true',
                      help='Whether to predict surface normals.')
  parser.add_argument('--strides', type=str, default='512,512',
                      help='Comma-separated string with strides of H and W.')
  parser.add_argument('--num_classes', type=int, default=21,
                      help='Number of classes to predict.')
  parser.add_argument('--ignore_label', type=int, default=255,
                      help='Index of label to ignore.')
  parser.add_argument('--restore_from', type=str, default=None,
                      help='Where restore model parameters from.')
  parser.add_argument('--save_dir', type=str, default='',
                      help='/path/to/save/predictions.')
  parser.add_argument('--colormap', type=str, default='',
                      help='/path/to/colormap/file.')

  return parser.parse_args()


def load(saver, sess, ckpt_path):
  """Loads the trained weights.
    
  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    ckpt_path: path to checkpoint file with parameters.
  """ 
  saver.restore(sess, ckpt_path)
  print('Restored model parameters from {}'.format(ckpt_path))


def parse_commastr(str_comma):
  """Read comma-sperated string.
  """
  if '' == str_comma:
    return None
  else:
    a, b =  map(int, str_comma.split(','))

  return [a,b]


def main():
  """Create the model and start the Inference process.
  """
  # Read CL arguments and snapshot the arguments into text file.
  args = get_arguments()
    
  # Parse image processing arguments.
  input_size = parse_commastr(args.input_size)
  strides = parse_commastr(args.strides)
  assert(input_size is not None and strides is not None)
  h, w = input_size

  # Create queue coordinator.
  coord = tf.train.Coordinator()

  # Load the data reader.
  with tf.name_scope('create_inputs'):
    reader = ImageReader(
        args.data_dir,
        args.data_list,
        None,
        False, # No random scale.
        False, # No random mirror.
        False, # No random crop, center crop instead
        args.ignore_label,
        IMG_MEAN)
    image = reader.datas[0]
    image_list = reader.file_lists[0]
  image_batch = tf.expand_dims(image, dim=0)

  # Create input tensor to the Network
  crop_image_batch = tf.placeholder(
      name='crop_image_batch',
      shape=[1,input_size[0],input_size[1],3],
      dtype=tf.float32)

  # Create network and output predictions.
  outputs = structured_predictor(crop_image_batch,
                                 args.num_classes,
                                 False,
                                 True)

  # Grab variable names which should be restored from checkpoints.
  restore_var = [
    v for v in tf.global_variables()
      if 'crop_image_batch' not in v.name]

  # Output Segmentation Predictions.
  segmentation_output = outputs[0]
  segmentation_output = tf.image.resize_bilinear(
      segmentation_output,
      tf.shape(crop_image_batch)[1:3,])
  segmentation_output = tf.nn.softmax(segmentation_output, dim=3)
  
  # Output Depth Estimations.
  depth_output = outputs[1]
  depth_output = tf.image.resize_bilinear(
      depth_output,
      tf.shape(crop_image_batch)[1:3,])

  # Output Surface Normal Estimations.
  normal_output = outputs[2]
  normal_output = tf.image.resize_bilinear(
      normal_output,
      tf.shape(crop_image_batch)[1:3,])

  # Set up tf session and initialize variables. 
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()
    
  sess.run(init)
  sess.run(tf.local_variables_initializer())
    
  # Load weights.
  if args.restore_from is not None:
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.restore_from)
    
  # Start queue threads.
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  # Get colormap.
  map_data = scipy.io.loadmat(args.colormap)
  key = os.path.basename(args.colormap).replace('.mat','')
  colormap = map_data[key]
  colormap *= 255
  colormap = colormap.astype(np.uint8)

  # Create directory for saving predictions.
  segmentation_dir = os.path.join(args.save_dir, 'segmentation_gray')
  segmentation_rgb_dir = os.path.join(args.save_dir, 'segmentation_color')
  depth_dir = os.path.join(args.save_dir, 'depth')
  normal_dir = os.path.join(args.save_dir, 'normal')
  if args.train_segmentation and not os.path.isdir(segmentation_dir):
    os.makedirs(segmentation_dir)
  if args.train_segmentation and not os.path.isdir(segmentation_rgb_dir):
    os.makedirs(segmentation_rgb_dir)
  if args.train_depth and not os.path.isdir(depth_dir):
    os.makedirs(depth_dir)
  if args.train_normal and not os.path.isdir(normal_dir):
    os.makedirs(normal_dir)
    
  # Iterate over inference steps.
  with open(args.data_list, 'r') as listf:
    num_steps = len(listf.read().split('\n'))-1

  for step in range(num_steps):
    img_batch = sess.run(image_batch)
    img_size = img_batch.shape
    padimg_size = list(img_size) # deep copy of img_size

    padimg_h, padimg_w = padimg_size[1:3]
    input_h, input_w = input_size

    if input_h > padimg_h:
      padimg_h = input_h
    if input_w > padimg_w:
      padimg_w = input_w
    # Update padded image size.
    padimg_size[1] = padimg_h
    padimg_size[2] = padimg_w
    padimg_batch = np.zeros(padimg_size, dtype=np.float32)
    img_h, img_w = img_size[1:3]
    padimg_batch[:, :img_h, :img_w, :] = img_batch

    # Create padded segmentation prediction array.
    segmentation_size = list(padimg_size)
    segmentation_size[-1] = args.num_classes
    segmentation_batch = np.zeros(segmentation_size, dtype=np.float32)
    segmentation_batch.fill(args.ignore_label)

    # Create padded depth estimation array.
    depth_size = list(padimg_size)
    depth_size[-1] = 1
    depth_batch = np.zeros(depth_size, dtype=np.float32)

    # Create padded surface normal estimation array.
    normal_size = list(padimg_size)
    normal_size[-1] = 3
    normal_batch = np.zeros(normal_size, dtype=np.float32)

    # Create padding array for recording number of patches.
    num_batch = np.zeros_like(depth_batch, dtype=np.float32)

    stride_h, stride_w = strides
    npatches_h = math.ceil(1.0*(padimg_h-input_h)/stride_h) + 1
    npatches_w = math.ceil(1.0*(padimg_w-input_w)/stride_w) + 1
    # Create the ending index of each patch
    patch_indh = np.linspace(input_h, padimg_h,
                             npatches_h, dtype=np.int32)
    patch_indw = np.linspace(input_w, padimg_w,
                             npatches_w, dtype=np.int32)

    for indh in patch_indh:
      for indw in patch_indw:
        sh, eh = indh - input_h, indh # start&end ind of H
        sw, ew = indw - input_w, indw # start&end ind of W
        cropimg_batch = padimg_batch[:, sh:eh, sw:ew, :]
        feed_dict = {crop_image_batch: cropimg_batch}

        outs = sess.run(
            [segmentation_output, depth_output, normal_output],
            feed_dict=feed_dict)
        segmentation_batch[:, sh:eh, sw:ew, :] += outs[0]
        depth_batch[:, sh:eh, sw:ew, :] += outs[1]
        normal_batch[:, sh:eh, sw:ew, :] += outs[2]
        num_batch[:, sh:eh, sw:ew, :] += 1

    num_batch = num_batch[0, :, :, :]

    # Discretize probability prediction to class index.
    segmentation_batch = segmentation_batch[0, :img_h, :img_w, :]
    segmentation_batch = np.argmax(segmentation_batch, axis=-1)
    segmentation_batch = segmentation_batch.astype(np.uint8)

    # Average the depth estimations.
    depth_batch = (depth_batch / num_batch * args.depth_unit).astype(np.int32)
    depth_batch = depth_batch[0, :img_h, :img_w, 0]

    # Average the surface normal estimations.
    normal_batch = (normal_batch / num_batch * 127.5 + 127.5).astype(np.uint8)
    normal_batch = normal_batch[0, :img_h, :img_w, :]

    basename = os.path.basename(image_list[step])
    basename = basename.replace('jpg', 'png')

    # Save segmentation and colorized results to files.
    if args.train_segmentation:
      segmentation_name = os.path.join(
          segmentation_dir,
          basename.replace('rgb', 'semantic'))
      Image.fromarray(segmentation_batch, mode='L').save(segmentation_name)

      segmentation_rgb_name = os.path.join(segmentation_rgb_dir,
                                           basename.replace('rgb', 'semantic'))
      color = colormap[segmentation_batch]
      Image.fromarray(color, mode='RGB').save(segmentation_rgb_name)

    # Save depth estimation results to files.
    if args.train_depth:
      depth_name = os.path.join(depth_dir, basename.replace('rgb', 'depth'))
      Image.fromarray(depth_batch, mode='I').save(depth_name)

    # Save surface normal estimation results to files.
    if args.train_normal:
      if 'area_6' in image_list[step]:
        normal_name = os.path.join(normal_dir, basename.replace('rgb', 'normal'))
      else:
        normal_name = os.path.join(normal_dir, basename.replace('rgb', 'normals'))
      Image.fromarray(normal_batch, mode='RGB').save(normal_name)

  coord.request_stop()
  coord.join(threads)
    

if __name__ == '__main__':
    main()
