"""Build Fully-Convolutional-Residual-Network (FCRN).
"""
import tensorflow as tf

from network.common.resnet_v1 import resnet_v1
import network.common.layers as nn


def conv(x,
         name,
         filters,
         kernel_sizes,
         strides,
         padding,
         relu=True,
         biased=True,
         bn=True,
         decay=0.99,
         is_training=True,
         use_global_status=True):
  """Convolutional layers with batch normalization and ReLU.
  """
  c_i = x.get_shape().as_list()[-1] # input channels
  c_o = filters # output channels

  # Define helper function.
  convolve = lambda i,k: tf.nn.conv2d(
      i,
      k,
      [1, strides, strides, 1],
      padding=padding)

  with tf.variable_scope(name) as scope:
    kernel = tf.get_variable(
        name='weights',
        shape=[kernel_sizes[0], kernel_sizes[1], c_i, c_o],
        trainable=is_training)

    if strides > 1:
      pad = kernel_size - 1
      pad_beg = pad // 2
      pad_end = pad - pad_beg
      pad_h = [pad_beg, pad_end]
      pad_w = [pad_beg, pad_end]
      x = tf.pad(x, [[0,0], pad_h, pad_w, [0,0]])

    output = convolve(x, kernel)

    # Add the biases.
    if biased:
      biases = tf.get_variable('biases',
                               [c_o],
                               trainable=is_training)
      output = tf.nn.bias_add(output, biases)

    # Apply batch normalization.
    if bn:
      is_bn_training = not use_global_status
      output = nn.batch_norm(output,
                             'BatchNorm',
                             is_training=is_bn_training,
                             decay=decay,
                             activation_fn=None)

    # Apply ReLU as activation function.
    if relu:
      output = tf.nn.relu(output)

  return output


def interleave(tensors, axis):
  old_shape = tensors[0].get_shape().as_list()[1:]
  new_shape = [-1] + old_shape
  new_shape[axis] *= len(tensors)
  return tf.reshape(tf.stack(tensors, axis+1), new_shape)


def unpool_as_conv(x, name, c_o, is_training=True,
                   use_global_status=True):
  """Model upconvolutions (unpooling + convolution) as
  interleaving feature maps of four convolutions (A,B,C,D).
  
  This function is the Building block for up-projections. 
  """
  with tf.variable_scope(name) as scope:
    # Convolution A (3x3)
    x_a = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
    x_a = conv(x_a, 'conv_a', c_o, [3, 3], 1, padding='VALID',
               relu=False, biased=True, bn=False, is_training=is_training)

    # Convolution B (2x3)
    x_b = tf.pad(x, [[0, 0], [1, 0], [1, 1], [0, 0]], mode='CONSTANT')
    x_b = conv(x_b, 'conv_b', c_o, [2, 3], 1, padding='VALID',
               relu=False, biased=True, bn=False, is_training=is_training)

    # Convolution C (3x2)
    x_c = tf.pad(x, [[0, 0], [1, 1], [1, 0], [0, 0]], mode='CONSTANT')
    x_c = conv(x_c, 'conv_c', c_o, [3, 2], 1, padding='VALID',
               relu=False, biased=True, bn=False, is_training=is_training)

    # Convolution D (2x2)
    x_d = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
    x_d = conv(x_d, 'conv_d', c_o, [2, 2], 1, padding='VALID',
               relu=False, biased=True, bn=False, is_training=is_training)


    # Interleaving elements of the four feature maps.
    x_l = interleave([x_a, x_b], axis=1)  # columns
    x_r = interleave([x_c, x_d], axis=1)  # columns
    output = interleave([x_l, x_r], axis=2) # rows
      
    is_bn_training = not use_global_status
    output = nn.batch_norm(output, 'BatchNorm',
                           is_training=is_bn_training,
                           decay=0.99,
                           activation_fn=None)
      
  return output


def up_project(x, name, c_o, is_training=True,
               use_global_status=False):
  """Up-Porjection depicted in: Deeper Depth Prediction with Fully
  Convolutional Residual Networks.
  """
  with tf.variable_scope(name):
    # Branch1.
    x1 = unpool_as_conv(
        x, 'br1/unpool', c_o,
        is_training=is_training,
        use_global_status=use_global_status)
    x1 = tf.nn.relu(x1)

    x1 = conv(x1, 'br1/conv1', c_o, [3, 3], 1, padding='SAME',
              biased=False, relu=False, bn=True, decay=0.99,
              is_training=is_training, use_global_status=use_global_status)

        
    # Branch 2
    x2 = unpool_as_conv(
        x, 'br2/unpool', c_o,
        is_training=is_training,
        use_global_status=use_global_status)
    
    # sum branches
    x = tf.nn.relu(x1+x2)

  return x


def _fcrn_builder(x,
                  name,
                  cnn_fn,
                  num_classes,
                  is_training,
                  use_global_status,
                  reuse=False):
  """Helper function to build FCRN model for depth estimation.
  """
  # Input size.
  h, w = x.get_shape().as_list()[1:3] # NxHxWxC

  # Build the base network.
  x = cnn_fn(x, name, is_training, use_global_status, reuse)

  with tf.variable_scope(name, reuse=reuse) as scope:
    with tf.variable_scope('block5') as scope:
      # Build Upsampling Porjection.
      x = conv(x, 'conv1', 1024, [1,1], 1, padding='SAME',
               biased=False, bn=True, relu=False, decay=0.99,
               is_training=is_training,
               use_global_status=use_global_status)

      x = up_project(x, 'upproj1', 512,
                     is_training, use_global_status)
      x = up_project(x, 'upproj2', 256,
                     is_training, use_global_status)
      x = up_project(x, 'upproj3', 128,
                     is_training, use_global_status)
      x = up_project(x, 'upproj4', 64,
                     is_training, use_global_status)

      seg = conv(x, 'fc1_seg', num_classes, [3,3], 1, padding='SAME',
                 biased=True, bn=False, relu=False,
                 is_training=is_training)

      dph = conv(x, 'fc1_depth', 1, [3,3], 1, padding='SAME',
                 biased=True, bn=False, relu=True,
                 is_training=is_training)

      nrm = conv(x, 'fc1_normal', 3, [3,3], 1, padding='SAME',
                 biased=True, bn=False, relu=False,
                 is_training=is_training)
      nrm = tf.nn.l2_normalize(nrm, dim=-1)

    return [seg, dph, nrm]


def resnet_v1_50(x,
                 name,
                 is_training,
                 use_global_status,
                 reuse=False):
  """Builds ResNet50 v1.
  """
  return resnet_v1(x,
                   name=name,
                   filters=[64,128,256,512],
                   num_blocks=[3,4,6,3],
                   strides=[2,2,2,1],
                   dilations=[None, None, None, None],
                   is_training=is_training,
                   use_global_status=use_global_status,
                   reuse=reuse)


def structured_predictor(x,
                         num_classes,
                         is_training,
                         use_global_status,
                         reuse=False):
  """Helper function to build FCRN as structured predictor.
  """
  scores = _fcrn_builder(x,
                         'resnet_v1_50',
                         resnet_v1_50,
                         num_classes,
                         is_training,
                         use_global_status,
                         reuse=reuse)

  return scores
