"""Define discriminator.
"""
import tensorflow as tf

import network.common.layers as nn


def _discriminator_builder(x,
                           name,
                           num_classes=40,
                           filters=[64,128,256,512,512],
                           num_blocks=[2,2,3,3,3],
                           strides=[2,2,2,2,1],
                           is_training=True,
                           use_global_status=True,
                           reuse=False):
  """Helper function to construct Discriminator.
  """
  if len(filters) != len(num_blocks)\
      or len(filters) != len(strides):
    raise ValueError('length of lists are not consistent')

  with tf.variable_scope(name, reuse=reuse) as scope:
    # blocks
    for ib in range(len(filters)):
      for iu in range(num_blocks[ib]):
        name_format = 'layer{:d}/conv{:d}_{:d}'
        block_name = name_format.format(ib+1, ib+1, iu+1)

        c_o = filters[ib] # output channel
        # strides at the end
        s = strides[ib] if strides[ib] else 1
        pad = 'VALID' if s > 1 else 'SAME'
        x = nn.conv(x,
                    name=block_name,
                    filters=c_o,
                    kernel_size=4,
                    strides=s,
                    padding=pad,
                    biased=False,
                    bn=True,
                    relu=True,
                    decay=0.99,
                    is_training=is_training,
                    use_global_status=use_global_status)

    x = nn.conv(x,
                name='block5/fc1_out',
                filters=num_classes,
                kernel_size=4,
                strides=1,
                padding='SAME',
                biased=True,
                bn=False,
                relu=False,
                is_training=is_training)
    print(x)

    return x


def discriminator(x,
                  num_classes,
                  is_training,
                  use_global_status,
                  reuse=False):
  return _discriminator_builder(x,
                                name='Discriminator',
                                num_classes=num_classes,
                                filters=[64,128,256,512,512],
                                num_blocks=[1,1,1,1,1],
                                strides=[2,2,2,2,1],
                                is_training=is_training,
                                use_global_status=use_global_status,
                                reuse=reuse)
