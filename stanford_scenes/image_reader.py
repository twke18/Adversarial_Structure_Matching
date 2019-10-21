"""Define data preprocessing functions.
"""
import os

import numpy as np
import tensorflow as tf


def image_scaling(img, label, depth, normal):
  """Randomly scales the images between 0.5 to 1.5 times the original size.

  Args:
    img: A tensor of size [height_in, width_in, 3]
    label: A tensor of size [height_in, width_in, 1]
    depth: A tensor of size [height_in, width_in, 1]
    normal: A tensor of size [height_in, width_in, 3]

  Returns:
    Two tensors of size [height_out, width_out, 3], and another
    three tensors of size [height_out, width_out, 1]
  """
  scale = tf.random_uniform(
      [1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
  h_new = tf.to_int32(tf.to_float(tf.shape(img)[0]) * scale)
  w_new = tf.to_int32(tf.to_float(tf.shape(img)[1]) * scale)
  new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
  # Rescale images by bilinear sampling.
  img = tf.image.resize_images(img,
                               new_shape,
                               tf.image.ResizeMethod.BILINEAR)
  # Rescale segmentations by nearest neighbor sampling.
  label = tf.image.resize_images(label,
                                 new_shape,
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # Rescale depths by bilinear sampling.
  depth = tf.image.resize_images(depth,
                                 new_shape,
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  depth /= scale
  # Rescale normals by nearest neighbor.
  normal = tf.image.resize_images(normal,
                                  new_shape,
                                  tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   
  return img, label, depth, normal


def image_mirroring(img, label, depth, normal):
  """Randomly horizontally mirrors the images and their labels.

  Args:
    img: A tensor of size [height_in, width_in, 3]
    label: A tensor of size [height_in, width_in, 1]
    depth: A tensor of size [height_in, width_in, 1]
    normal: A tensor of size [height_in, width_in, 3]

  Returns:
    Two tensor of size [height_in, width_in, 3], and another
    three tensors of size [height_in, width_in, 1]
  """
  distort_left_right_random = tf.random_uniform(
      [1], 0, 1.0, dtype=tf.float32)
  distort_left_right_random = distort_left_right_random[0]

  mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
  mirror = tf.boolean_mask([0, 1, 2], mirror)
  img = tf.reverse(img, mirror)
  label = tf.reverse(label, mirror)
  depth = tf.reverse(depth, mirror)
  normal = tf.reverse(normal, mirror)
  # Horizontal flipping the x-direction vector value.
  normal_mirror = tf.less(
      tf.stack([distort_left_right_random, 1.0, 1.0]),
      0.5)
  normal_mirror = tf.to_int32(normal_mirror)
  normal_mirror = tf.gather(tf.stack([1.0, -1.0]), normal_mirror)
  normal *= tf.cast(normal_mirror, dtype=tf.float32)

  return img, label, depth, normal


def crop_and_pad_image_and_labels(img,
                                  label,
                                  depth,
                                  normal,
                                  crop_h,
                                  crop_w,
                                  ignore_label=255,
                                  random_crop=True):
  """Randomly crops and pads the images and their labels.

  Args:
    img: A tensor of size [batch_size, height_in, width_in, channels]
    label: A tensor of size [batch_size, height_in, width_in]
    crop_h: A number indicating the height of output data.
    crop_w: A number indicating the width of output data.
    ignore_label: A number indicating the indices of ignored label.
    random_crop: enable/disable random_crop for random cropping.

  Returns:
    A tensor of size [batch_size, height_out, width_out, channels], and another
    tensor of size [batch_size, height_out, width_out, 1]
  """
  # Needs to be subtracted and later added due to 0 padding.
  label = tf.cast(label, dtype=tf.float32)
  label = label - ignore_label 

  # Concatenate images with labels, which makes random cropping easier.
  combined = tf.concat(values=[img, label, depth, normal], axis=-1) 
  combined_shape = tf.shape(combined)
  combined = tf.image.pad_to_bounding_box(
      combined,
      0,
      0,
      tf.maximum(crop_h, combined_shape[0]),
      tf.maximum(crop_w, combined_shape[1]))

  if random_crop:
    channels = combined.get_shape().as_list()[-1]
    combined = tf.random_crop(combined, [crop_h,crop_w,channels])
  else:
    combined= tf.image.resize_image_with_crop_or_pad(combined, crop_h, crop_w)

  img = combined[:, :, :3]
  label = combined[:, :, 3:4]
  label = label + ignore_label
  label = tf.cast(label, dtype=tf.uint8)
  depth = combined[:, :, 4:5]
  normal = combined[:, :, 5:8]
    
  # Set static shape so that tensorflow knows shape at running. 
  img.set_shape((crop_h, crop_w, 3))
  label.set_shape((crop_h, crop_w, 1))
  depth.set_shape((crop_h, crop_w, 1))
  normal.set_shape((crop_h, crop_w, 3))

  return img, label, depth, normal


def read_labeled_image_list(data_dir, data_list):
  """Reads txt file containing paths to images and ground truth masks.
    
  Args:
    data_dir: A string indicating the path to the root directory of images
      and masks.
    data_list: A string indicating the path to the file with lines of the form
      '/path/to/image /path/to/label'.
       
  Returns:
    Five lists with all file names for images, segmentations, depths, normals
    and validity masks, respectively.
  """
  f = open(data_list, 'r')
  images = []
  segmentations = []
  depths = []
  normals = []
  for line in f:
    line = line.strip("\n")
    images.append(
        os.path.join(data_dir, line.format('rgb', 'rgb')))
    segmentations.append(
        #os.path.join(data_dir, line.format('segcls', 'semantic')))
        os.path.join(data_dir, line.format('depth', 'depth')))
    depths.append(
        os.path.join(data_dir, line.format('depth', 'depth')))
    if 'area_6' in line:
      normals.append(
          os.path.join(data_dir, line.format('normal', 'normal')))
    else:
      normals.append(
          os.path.join(data_dir, line.format('normal', 'normals')))
  return images, segmentations, depths, normals


def read_images_from_disk(input_queue,
                          input_size,
                          random_scale,
                          random_mirror,
                          random_crop,
                          ignore_label,
                          img_mean):
  """Reads one image and its corresponding label and perform pre-processing.
    
  Args:
    input_queue: A tensorflow queue with paths to the image and its
      segmentation, depth, surface normal.
    input_size: A tuple with entries of height and width. If None, return
      images of original size.
    random_scale: enable/disable random_scale for randomly scaling images
      and their labels.
    random_mirror: enable/disable random_mirror for randomly and horizontally
      flipping images and their labels.
    ignore_label: A number indicating the index of label to ignore.
    img_mean: A vector indicating the mean colour values of RGB channels.
      
  Returns:
    Five tensors: the decoded image and its segmentation, depth, surface normal.
  """

  contents = [tf.read_file(c) for c in input_queue]
    
  img = tf.image.decode_jpeg(contents[0], channels=3)
  img = tf.cast(img, dtype=tf.float32)
  # Extract mean.
  img -= img_mean

  label = tf.image.decode_png(contents[1], channels=1)
  depth = tf.image.decode_png(contents[2], channels=1, dtype=tf.uint16)
  depth = tf.cast(depth, dtype=tf.int16) + 1 # make invalid depth value to 0
  depth = tf.cast(depth, dtype=tf.float32) - 1 # make invalid depth value to -1
  normal = tf.image.decode_png(contents[3], channels=3)
  normal = tf.cast(normal, dtype=tf.float32)
  normal -= 127.5

  if input_size is not None:
    h, w = input_size

    # Randomly scale the images and labels.
    if random_scale:
      img, label, depth, normal = image_scaling(img, label, depth, normal)

    # Randomly mirror the images and labels.
    if random_mirror:
      img, label, depth, normal = image_mirroring(img, label, depth, normal)

    # Randomly crops the images and labels.
    img, label, depth, normal = crop_and_pad_image_and_labels(
        img, label, depth, normal, h, w, ignore_label, random_crop)


  return img, label, depth, normal


class ImageReader(object):
  """
  Generic ImageReader which reads images and corresponding
  segmentation masks from the disk, and enqueues them into
  a TensorFlow queue.
  """

  def __init__(self,
               data_dir,
               data_list,
               input_size,
               random_scale,
               random_mirror,
               random_crop,
               ignore_label,
               img_mean):
    """
    Initialise an ImageReader.
          
    Args:
      data_dir: path to the directory with images, segmentations, depths,
        surface normals and validity maps.
      data_list: path to the file with lines of the form
                 '/path/to/image /path/to/mask'.
      input_size: a tuple with (height, width) values, to which all the
                  images will be resized.
      random_scale: whether to randomly scale the images.
      random_mirror: whether to randomly mirror the images.
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      A tensor of size [batch_size, height_out, width_out, channels], and
      another tensor of size [batch_size, height_out, width_out]
    """
    self.data_dir = data_dir
    self.data_list = data_list
    self.input_size = input_size
          
    self.file_lists = read_labeled_image_list(
        self.data_dir, self.data_list)
    file_lists = [tf.convert_to_tensor(f, dtype=tf.string)
                    for f in self.file_lists]
    self.queue = tf.train.slice_input_producer(
        file_lists,
        shuffle=input_size is not None) # not shuffling if it is val
    self.datas = read_images_from_disk(
        self.queue,
        self.input_size,
        random_scale,
        random_mirror,
        random_crop,
        ignore_label,
        img_mean) 


  def dequeue(self, num_elements):
    """Packs images and labels into a batch.
        
    Args:
      num_elements: A number indicating the batch size.
          
    Returns:
      Two tensors of size [batch_size, height_out, width_out, 3], and
      three tensors of size [batch_size, height_out, width_out, 1]
    """
    # images, segmentations, depths, normals.
    datas_batch = tf.train.batch(
        self.datas,
        num_elements,
        num_threads=2)
    return datas_batch
