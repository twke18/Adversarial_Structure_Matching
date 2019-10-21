"""Define analyzer.
"""
import tensorflow as tf

import network.common.layers as nn


def _unet_builder(x,
                  mask,
                  name,
                  filters=[64,128,256,512,1024],
                  num_blocks=[2,3,3,3,3],
                  strides=[2,2,2,2,2],
                  is_training=True,
                  use_global_status=False,
                  reuse=False):
  """Helper function to construct UNet.
  """
  if len(filters) != len(num_blocks)\
      or len(filters) != len(strides):
    raise ValueError('length of lists are not consistent')

  with tf.variable_scope('Analyzer', reuse=reuse) as scope:
    with tf.name_scope(name):
      input_x = x

      # Encoder.
      shortcuts = []
      not_ignore_masks = []
      for ib in range(len(filters)):
        for iu in range(num_blocks[ib]):
          name_format = 'layer{:d}/unit_{:d}/encoder/'
          block_name = name_format.format(ib+1, iu+1)
          c_o = filters[ib] # output channel

          # strides at the begginning
          s = strides[ib] if iu == 0 else 1
          padding = 'VALID' if s > 1 else 'SAME'
          if ib == 0 and iu == 0:
            x = []
            for ix,in_x in enumerate(input_x):
              x.append(nn.conv(in_x,
                               name=block_name+'conv{:d}'.format(ix),
                               filters=int(c_o/2),
                               #filters=c_o,
                               kernel_size=3,
                               strides=s,
                               padding=padding,
                               #biased=False,
                               #bn=True,
                               biased=True,
                               bn=False,
                               relu=False,
                               decay=0.99,
                               is_training=is_training,
                               use_global_status=use_global_status))
            x = tf.concat(x, axis=-1, name=block_name+'concat')
          else:
            x = nn.conv(x,
                        name=block_name+'conv',
                        filters=c_o,
                        kernel_size=3,
                        strides=s,
                        padding=padding,
                        #biased=False,
                        #bn=True,
                        biased=True,
                        bn=False,
                        relu=False,
                        decay=0.99,
                        is_training=is_training,
                        use_global_status=use_global_status)

          if iu == 0:
            mask = nn.max_pool(mask,
                               block_name+'mask_pool',
                               3,
                               s,
                               padding=padding)
            not_ignore_masks.append(1-mask)
          f = tf.multiply(x,
                          not_ignore_masks[-1],
                          name=block_name+'masked_conv')
          tf.add_to_collection('Analyzer/features', f)
          x = tf.nn.relu(x)
        print(x)
        shortcuts.append(x)

      # Decoder.
      for ib in range(len(shortcuts)-1, 0 ,-1):
        for iu in range(num_blocks[ib-1]):
          n, h, w, c_o = shortcuts[ib-1].get_shape().as_list()
          name_format = 'layer{:d}/unit_{:d}/decoder/'
          block_name = name_format.format(2*len(filters)-ib, iu+1)
          x = nn.conv(x,
                      name=block_name+'conv',
                      filters=c_o,
                      kernel_size=3,
                      strides=1,
                      padding='SAME',
                      #biased=False,
                      #bn=True,
                      biased=True,
                      bn=False,
                      relu=False,
                      decay=0.99,
                      is_training=is_training,
                      use_global_status=use_global_status)

          f = tf.multiply(x,
                          not_ignore_masks[ib],
                          name=block_name+'masked_conv')
          tf.add_to_collection('Analyzer/features', f)
          x = tf.nn.relu(x)
          if iu == 0:
            x = tf.image.resize_bilinear(x, [h,w])
            x = tf.concat([x, shortcuts[ib-1]], axis=-1)
        print(x)

      c_i = 0
      for in_x in input_x:
        c_i += in_x.get_shape().as_list()[-1]
      x = nn.conv(x,
                  name='block5/fc',
                  filters=c_i,
                  kernel_size=1,
                  strides=1,
                  padding='SAME',
                  biased=True,
                  bn=False,
                  relu=False,
                  is_training=is_training)
      x = tf.image.resize_bilinear(x, tf.shape(input_x[0])[1:3])
      tf.add_to_collection('Analyzer/outputs', x)
    return x


def analyzer(x, mask, name, is_training=True,
          use_global_status=False, reuse=False):
  """Build UNet.
  """
  score = _unet_builder(x,
                        mask,
                        name=name,
                        filters=[32,64,128,128,128],
                        num_blocks=[1]*5,
                        strides=[2]*5,
                        is_training=is_training,
                        use_global_status=use_global_status,
                        reuse=reuse)

  return score
