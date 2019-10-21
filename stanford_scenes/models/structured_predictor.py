"""Define U-Net.
"""
import tensorflow as tf

import network.common.layers as nn


def _unet_builder(x,
                  name,
                  filters=[64,128,256,512,1024],
                  num_blocks=[2,3,3,3,3],
                  strides=[2,2,2,2,2],
                  dilations=[None,None,None,None,None],
                  num_classes=40,
                  is_training=True,
                  use_global_status=False,
                  reuse=False):
  """Helper function to construct UNet.
  """
  if len(filters) != len(num_blocks)\
      or len(filters) != len(strides):
    raise ValueError('length of lists are not consistent')

  with tf.variable_scope(name, reuse=reuse) as scope:
    # Encoder.
    shortcuts = []
    for ib in range(len(filters)):
      for iu in range(num_blocks[ib]):
        name_format = 'layer{:d}/unit_{:d}/encoder/'
        block_name = name_format.format(ib+1, iu+1)
        c_o = filters[ib] # output channel

        # strides at the begginning
        s = strides[ib] if iu == 0 else 1
        d = dilations[ib]
        if d is not None and d > 1 and s == 1:
          x = nn.atrous_conv(x,
                             name=block_name+'/conv',
                             filters=c_o,
                             kernel_size=3,
                             dilation=d,
                             padding='SAME',
                             biased=False,
                             bn=True,
                             relu=True,
                             decay=0.99,
                             is_training=is_training,
                             use_global_status=use_global_status)
        else:
          padding = 'VALID' if s > 1 else 'SAME'
          ksize = s*2 if s > 1 else 3
          x = nn.conv(x,
                      name=block_name+'/conv',
                      filters=c_o,
                      kernel_size=ksize,
                      strides=s,
                      padding=padding,
                      biased=False,
                      bn=True,
                      relu=True,
                      decay=0.99,
                      is_training=is_training,
                      use_global_status=use_global_status)
      print(x)
      shortcuts.append(x)

    # Decoder.
    for ib in range(len(shortcuts)-1, 0 ,-1):
      #for iu in range(num_blocks[ib-1]):
      for iu in range(3):
        n, h, w, c_o = shortcuts[ib-1].get_shape().as_list()
        name_format = 'layer{:d}/unit_{:d}/decoder/'
        block_name = name_format.format(2*len(filters)-ib, iu+1)
        x = nn.conv(x,
                    name=block_name+'conv',
                    filters=c_o,
                    kernel_size=3,
                    strides=1,
                    padding='SAME',
                    biased=False,
                    bn=True,
                    relu=True,
                    decay=0.99,
                    is_training=is_training,
                    use_global_status=use_global_status)
        if iu == 0:
          x = tf.image.resize_bilinear(x, [h,w])
          x = tf.concat([x, shortcuts[ib-1]], axis=-1)
      print(x)

    # output segmentation, depth and surface normal estimation.
    block_name = 'block5'
    seg = nn.conv(x, block_name+'/fc1_seg', num_classes, 3, 1, padding='SAME',
                  biased=True, bn=False, relu=False, is_training=is_training)

    dph = nn.conv(x, block_name+'/fc1_depth', 1, 3, 1, padding='SAME',
                  biased=True, bn=False, relu=True, is_training=is_training)

    nrm = nn.conv(x, block_name+'/fc1_normal', 3, 3, 1, padding='SAME',
                  biased=True, bn=False, relu=False, is_training=is_training)
    nrm = tf.nn.l2_normalize(nrm, dim=-1)

    return [seg, dph, nrm]


def structured_predictor(x,
                         num_classes=40,
                         is_training=True,
                         use_global_status=False,
                         reuse=False):
  """Build UNet as structured predictor.
  """
  scores = _unet_builder(x,
                         name='unet_512',
                         filters=[64,128,256,512,512,512,512,512,512],
                         #num_blocks=[3]*9,
                         num_blocks=[1]*9,
                         strides=[2]*9,
                         dilations=[None]*9,
                         num_classes=num_classes,
                         is_training=is_training,
                         use_global_status=use_global_status,
                         reuse=reuse)

  return scores
