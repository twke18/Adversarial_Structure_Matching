"""Training scripts for ASM structured predictor on 2D3DS dataset.
"""
from __future__ import print_function

import argparse
import os
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from stanford_scenes.models.structured_predictor import structured_predictor
from stanford_scenes.models.analyzer import analyzer
from stanford_scenes.image_reader import ImageReader
import network.common.layers as nn
import utils.general

IMG_MEAN = np.array((122.675, 116.669, 104.008), dtype=np.float32)


def get_arguments():
  """Parse all the arguments provided from the CLI.
    
  Returns:
    A list of parsed arguments.
  """
  parser = argparse.ArgumentParser(description='Semantic Segmentation')
  # Data parameters
  parser.add_argument('--batch_size', type=int, default=1,
                      help='Number of images in one step.')
  parser.add_argument('--data_dir', type=str, default='',
                      help='/path/to/dataset/.')
  parser.add_argument('--data_list', type=str, default='',
                      help='/path/to/datalist/file.')
  parser.add_argument('--ignore_label', type=int, default=255,
                      help='The index of the label to ignore.')
  parser.add_argument('--input_size', type=str, default='336,336',
                      help='Comma-separated string with H and W of image.')
  parser.add_argument('--depth_unit', type=float, default=512.0,
                      help='Each pixel value difference means 1/depth_unit '
                           'meters.')
  parser.add_argument('--train_segmentation', action='store_true',
                      help='Whether to train for semantic segmentation.')
  parser.add_argument('--train_depth', action='store_true',
                      help='Whether to train for depth estimation.')
  parser.add_argument('--train_normal', action='store_true',
                      help='Whether to train for surface normals prediction.')
  parser.add_argument('--adv_lambda', type=float, default=0.01,
                      help='Weighting for adversarial loss of Analyzer.')
  parser.add_argument('--rec_lambda', type=float, default=10.0,
                      help='Weighting for reconstruction loss of Analyzer.')
  parser.add_argument('--sup_lambda', type=float, default=1.0,
                      help='Weighting paramters for supervising loss.')
  parser.add_argument('--loss_on_structured_predictor', action='store_true',
                      help='Whether impose loss on structured predictor.')
  # Training paramters
  parser.add_argument('--is_training', action='store_true',
                      help='Whether to updates weights.')
  parser.add_argument('--use_global_status', action='store_true',
                      help='Whether to updates moving mean and variance.')
  parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                      help='Base learning rate.')
  parser.add_argument('--power', type=float, default=0.9,
                      help='Decay for poly learing rate policy.')
  parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum component of the optimiser.')
  parser.add_argument('--weight_decay', type=float, default=5e-4,
                      help='Regularisation parameter for L2-loss.')
  parser.add_argument('--num_classes', type=int, default=50,
                      help='Number of classes to predict.')
  parser.add_argument('--num_steps', type=int, default=20000,
                      help='Number of training steps.')
  parser.add_argument('--iter_size', type=int, default=10,
                      help='Number of iteration to update weights')
  parser.add_argument('--random_mirror', action='store_true',
                      help='Whether to randomly mirror the inputs.')
  parser.add_argument('--random_crop', action='store_true',
                      help='Whether to randomly crop the inputs.')
  parser.add_argument('--random_scale', action='store_true',
                      help='Whether to randomly scale the inputs.')
  parser.add_argument('--upscale_predictions', action='store_true',
                      help='Whether to upscale resolution of predictions.')
  # Misc paramters
  parser.add_argument('--restore_from', type=str, default=None,
                      help='Where restore model parameters from.')
  parser.add_argument('--save_pred_every', type=int, default=10000,
                      help='Save summaries and checkpoint every often.')
  parser.add_argument('--update_tb_every', type=int, default=200,
                      help='Update Tensorboard summaries every often.')
  parser.add_argument('--snapshot_dir', type=str, default='',
                      help='Where to save snapshots of the model.')
  parser.add_argument('--not_restore_classifier', action='store_true',
                      help='Whether to not restore classifier layers.')

  return parser.parse_args()


def save(saver, sess, logdir, step):
  """Saves the trained weights.
   
  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    logdir: path to the snapshots directory.
    step: current training step.
  """
  model_name = 'model.ckpt'
  checkpoint_path = os.path.join(logdir, model_name)
    
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  saver.save(sess, checkpoint_path, global_step=step)
  print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
  """Loads the trained weights.
    
  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    ckpt_path: path to checkpoint file with parameters.
  """ 
  saver.restore(sess, ckpt_path)
  print('Restored model parameters from {}'.format(ckpt_path))


def main():
  """Create the model and start training.
  """
  # Read CL arguments and snapshot the arguments into text file.
  args = get_arguments()
  utils.general.snapshot_arg(args)
    
  # The input size.
  h, w = map(int, args.input_size.split(','))
  input_size = (h, w)
    
  # Create queue coordinator.
  coord = tf.train.Coordinator()

  # current step
  step_ph = tf.placeholder(dtype=tf.float32, shape=())

  # Load the data reader.
  with tf.device('/cpu:0'):
    with tf.name_scope('create_inputs'):
      reader = ImageReader(
          args.data_dir,
          args.data_list,
          input_size,
          args.random_scale,
          args.random_mirror,
          args.random_crop,
          args.ignore_label,
          IMG_MEAN)

      datas_batch = reader.dequeue(args.batch_size)
      datas_batch[2] = datas_batch[2] / args.depth_unit
      image_batch = tf.placeholder(datas_batch[0].dtype,
                                   datas_batch[0].shape)
      label_batch = tf.placeholder(datas_batch[1].dtype,
                                   datas_batch[1].shape)
      depth_batch = tf.placeholder(datas_batch[2].dtype,
                                   datas_batch[2].shape)
      normal_batch = tf.placeholder(datas_batch[3].dtype,
                                    datas_batch[3].shape)

  # Create network and predictions.
  outputs = structured_predictor(image_batch,
                                 args.num_classes,
                                 args.is_training,
                                 args.use_global_status)

  # Either up-sample predictions or down-sample ground-truths.
  if args.upscale_predictions:
    for output_index, output in enumerate(outputs):
      out_h, out_w = output.get_shape().as_list()[1:3]
      if out_h != h or out_w != w:
        outputs[output_index] = tf.image.resize_bilinear(
            output, [h,w])
    images = image_batch
    labels = label_batch
    depths = depth_batch
    normals = normal_batch
  else:
    images = tf.image.resize_bilinear(
        image_batch, outputs[0].get_shape().as_list()[1:3])
    labels = tf.image.resize_nearest_neighbor(
        label_batch, outputs[0].get_shape().as_list()[1:3])
    depths = tf.image.resize_nearest_neighbor(
        depth_batch, outputs[1].get_shape().as_list()[1:3])
    normals = tf.image.resize_nearest_neighbor(
        normal_batch, outputs[2].get_shape().as_list()[1:3])

  # Ignore the location where the label value is larger than args.num_classes.
  labels_flat = tf.reshape(labels, (-1,))
  not_ignore_labels = tf.less_equal(labels_flat, args.num_classes-1)

  # Ignore the location where the depth value <= 0
  depths_flat = tf.reshape(depths, (-1,))
  not_ignore_depths = tf.greater(depths_flat, 0.0)

  # Ignore the location where the normal value != [128,128,128].
  # The normal is centered at 127.5 in ImageReader.
  ignore_normal_yz = tf.reduce_all(
      tf.equal(normals[:,:,:,1:], 0.5),
      axis=-1)
  ignore_normal_x = tf.equal(tf.abs(normals[:,:,:,0]), 0.5)
  ignore_normals = tf.logical_and(ignore_normal_yz,
                                  ignore_normal_x)
  ignore_normals = tf.reshape(ignore_normals, (-1,))
  not_ignore_normals = tf.logical_not(ignore_normals)
  normals = tf.nn.l2_normalize(normals, dim=-1)
  normals_flat = tf.reshape(normals, (-1,3))

  # Extract the indices of labels where the gradients are propogated.
  valid_label_inds = tf.squeeze(tf.where(not_ignore_labels), 1)
  valid_depth_inds = tf.squeeze(tf.where(not_ignore_depths), 1)
  valid_normal_inds = tf.squeeze(tf.where(not_ignore_normals), 1)

  # Grab variable names which should be restored from checkpoints.
  restore_var = [
    v for v in tf.global_variables()
      if 'block5' not in v.name or not args.not_restore_classifier]

  # Prepare inputs for Analyzer.
  gts, negs, preds, split_inds = [], [], [], [0,0,0]
  if args.train_segmentation:
    gts.append(tf.one_hot(tf.squeeze(labels, axis=-1),
                          depth=args.num_classes))
    negs.append(tf.one_hot(tf.argmax(outputs[0], axis=-1),
                           depth=args.num_classes))
    preds.append(tf.nn.softmax(outputs[0], dim=-1))
    split_inds[0] = args.num_classes

  if args.train_depth:
    gts.append(tf.log(tf.maximum(depths, 1e-2)))
    log_out = tf.log(tf.maximum(outputs[1], 1e-2))
    negs.append(log_out)
    preds.append(log_out)
    split_inds[1] = 1

  if args.train_normal:
    gts.append(normals)
    negs.append(outputs[2])
    preds.append(outputs[2])
    split_inds[2] = 3

  not_ignore_mask = tf.logical_and(not_ignore_depths,
                                   not_ignore_normals)
  not_ignore_mask = tf.reshape(not_ignore_mask,
                               tf.shape(depths)[:3])
  not_ignore_mask = tf.expand_dims(not_ignore_mask, axis=-1)
  not_ignore_mask = tf.cast(not_ignore_mask, dtype=tf.float32)
  ignore_mask = 1 - not_ignore_mask
  analyzer(gts, ignore_mask, 'gts', True, False, False)
  analyzer(negs, ignore_mask, 'negs', True, False, True)
  analyzer(preds, ignore_mask, 'preds', True, False, True)

  gt_recs = [v for v in tf.get_collection('Analyzer/outputs')
               if 'gts' in v.name][0]
  gt_recs = tf.split(gt_recs, split_inds, axis=-1)
  pred_recs = [v for v in tf.get_collection('Analyzer/outputs')
                if 'preds' in v.name][0]
  pred_recs = tf.split(pred_recs, split_inds, axis=-1)

  # Define losses.
  analyzer_adv_losses, analyzer_rec_losses, match_losses = [], [], []
  reduced_losses = []
  # Define segmentation loss and prediction errors.
  if args.train_segmentation:
    # Segmentation loss of prediction.
    labels_gather = tf.to_int32(tf.gather(labels_flat, valid_label_inds))
    segmentation_output = tf.reshape(outputs[0],
                                     [-1, args.num_classes])
    segmentation_output = tf.gather(segmentation_output,
                                    valid_label_inds)
    segmentation_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=segmentation_output,
        labels=labels_gather)
    segmentation_loss = tf.reduce_mean(segmentation_loss)

    if args.loss_on_structured_predictor:
      reduced_losses.append(segmentation_loss)

    # ASML Loss (Reconstruction segmentation).
    rec_segmentation_output = tf.reshape(gt_recs[0],
                                         [-1, args.num_classes])
    rec_segmentation_output = tf.gather(rec_segmentation_output,
                                        valid_label_inds)
    rec_segmentation_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=rec_segmentation_output,
        labels=labels_gather)
    rec_segmentation_loss = tf.reduce_mean(rec_segmentation_loss)
    analyzer_rec_losses.append(rec_segmentation_loss)

  # Define depth loss and prediction errors.
  if args.train_depth:
    # Depth loss of prediction.
    depths_gather = tf.gather(depths_flat, valid_depth_inds)
    depth_output = tf.gather(tf.reshape(outputs[1], [-1,]),
                             valid_depth_inds)
    depth_diff = depth_output - depths_gather
    depth_loss = tf.reduce_mean(depth_diff**2)
    depth_absrel = tf.reduce_sum(
        tf.abs(depth_diff) / depths_gather)
    depth_absrel /= args.batch_size

    if args.loss_on_structured_predictor:
      reduced_losses.append(depth_loss)

    # Reconstruction loss.
    rec_depth_output = tf.nn.relu(gt_recs[1])
    rec_depth_output = tf.gather(
        tf.reshape(rec_depth_output, [-1,]),
        valid_depth_inds)
    rec_depth_loss = tf.reduce_mean(
        (rec_depth_output - depths_gather)**2)
    analyzer_rec_losses.append(rec_depth_loss)

  # Define surface normal loss and prediction errors.
  if args.train_normal:
    # Surface normal loss of prediction.
    normals_gather = tf.gather(normals_flat, valid_normal_inds)
    normal_output = tf.gather(tf.reshape(outputs[2], [-1, 3]),
                              valid_normal_inds)
    normal_loss = -tf.reduce_mean(tf.reduce_sum(
      normal_output * normals_gather, axis=-1))
    normal_loss *= 10.0

    if args.loss_on_structured_predictor:
      reduced_losses.append(normal_loss)

    # Reconstruction loss.
    rec_normal_output = tf.gather(tf.reshape(gt_recs[2], [-1, 3]),
                                  valid_normal_inds)
    rec_normal_output = tf.nn.l2_normalize(rec_normal_output,
                                           dim=-1)
    rec_normal_loss = -tf.reduce_mean(tf.reduce_sum(
        rec_normal_output * normals_gather, axis=-1))
    analyzer_rec_losses.append(rec_normal_loss)

  # Compute adversarial loss for analyzer and matching loss for predictor.
  feat_collections = tf.get_collection('Analyzer/features')
  print(feat_collections)
  feat_names = ['layer1', 'layer2']
  for feat_name in feat_names:
    pred = [v for v in feat_collections
              if feat_name in v.name and 'preds' in v.name][0]
    gt = [v for v in feat_collections
            if feat_name in v.name and 'gts' in v.name][0]
    neg = [v for v in feat_collections
             if feat_name in v.name and 'negs' in v.name][0]
    pred = tf.nn.l2_normalize(pred, dim=-1)
    gt = tf.nn.l2_normalize(gt, dim=-1)
    neg = tf.nn.l2_normalize(neg, dim=-1)
    match_loss = tf.reduce_sum(tf.squared_difference(gt, pred),
                               axis=-1)
    adv_loss = tf.reduce_sum(tf.squared_difference(gt, neg),
                             axis=-1)
    adv_loss = tf.maximum(0.0, 0.80 - adv_loss)
    match_loss = tf.reduce_mean(match_loss)
    adv_loss = tf.reduce_mean(adv_loss)

    match_losses.append(match_loss)
    analyzer_adv_losses.append(adv_loss)

  # Define weight regularization loss.
  w = args.weight_decay
  l2_losses = [w*tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if 'weights' in v.name and 'Analyzer' not in v.name]
  analyzer_l2_losses = [
    w*tf.nn.l2_loss(v) for v in tf.trainable_variables()
      if 'weights' in v.name and 'Analyzer' in v.name]

  # Sum losses on Structured Predictor.
  if args.loss_on_structured_predictor:
    reduced_losses = [l*args.sup_lambda for l in reduced_losses]
  reduced_losses.append(tf.add_n(match_losses))
  reduced_losses.append(tf.add_n(l2_losses))
  reduced_loss = tf.add_n(reduced_losses)

  # Sum losses on Analyzer.
  dec = tf.pow(10.0, -step_ph/args.num_steps) # 1~0.1
  analyzer_max_loss = tf.add_n(analyzer_adv_losses) * args.adv_lambda * dec
  analyzer_min_loss = tf.add_n(analyzer_rec_losses) * args.rec_lambda
  analyzer_min_loss += tf.add_n(analyzer_l2_losses)

  # Grab variable names which are used for training.
  all_trainable = tf.trainable_variables()
  pred_trainable = [
    v for v in all_trainable
      if 'block5' in v.name and 'Analyzer' not in v.name]
  base_trainable = [
    v for v in all_trainable
      if 'block5' not in v.name and 'Analyzer' not in v.name]
  analyzer_trainable = [
    v for v in tf.trainable_variables() if 'Analyzer' in v.name]

  # Computes gradients per iteration.
  grads = tf.gradients(reduced_loss,
                       base_trainable + pred_trainable)
  grads_base = grads[:len(base_trainable)]
  grads_pred = grads[len(base_trainable):]
  grads_analyzer_max = [
    g if g is not None else None
      for g in tf.gradients(analyzer_max_loss, analyzer_trainable)]
  grads_analyzer_min = tf.gradients(analyzer_min_loss,
                                    analyzer_trainable)

  # Define optimisation parameters.
  base_lr = tf.constant(args.learning_rate)
  learning_rate = tf.scalar_mul(
      base_lr,
      tf.pow((1-step_ph/args.num_steps), args.power))

  opt_base = tf.train.MomentumOptimizer(learning_rate*1.0,
                                        args.momentum)
  opt_pred = tf.train.MomentumOptimizer(learning_rate*1.0,
                                        args.momentum)
  opt_analyzer_max = tf.train.MomentumOptimizer(learning_rate,
                                                args.momentum)
  opt_analyzer_min = tf.train.MomentumOptimizer(learning_rate,
                                                args.momentum)

  # Define tensorflow operations which apply gradients to update variables.
  train_op_base = opt_base.apply_gradients(
      zip(grads_base, base_trainable))
  train_op_pred = opt_pred.apply_gradients(
      zip(grads_pred, pred_trainable))
  train_op_analyzer_max = opt_analyzer_max.apply_gradients(
      zip(grads_analyzer_max, analyzer_trainable))
  train_op_analyzer_min = opt_analyzer_min.apply_gradients(
      zip(grads_analyzer_min, analyzer_trainable))

  train_op_s = tf.group(train_op_base, train_op_pred)
  train_op_a = tf.group(train_op_analyzer_max,
                        train_op_analyzer_min)

  # Process for visualisation.
  with tf.device('/cpu:0'):
    # Image summary for input image, ground-truth label and prediction.
    output_vis = []
    in_summary = tf.py_func(
        utils.general.inv_preprocess,
        [image_batch, IMG_MEAN],
        tf.uint8)
    output_vis.append(in_summary)

    if args.train_segmentation:
      # Visualize segmentation ground-truths.
      labels_vis = tf.cast(label_batch, dtype=tf.uint8)
      lab_summary = tf.py_func(
          utils.general.decode_labels,
          [labels_vis, args.num_classes],
          tf.uint8)
      output_vis.append(lab_summary)
      # Visualize segmentation predictions.
      segmentation_vis = tf.image.resize_images(
          outputs[0],
          tf.shape(image_batch)[1:3,],
          tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      segmentation_vis = tf.argmax(segmentation_vis, axis=3)
      segmentation_vis = tf.expand_dims(segmentation_vis, dim=3)
      segmentation_vis = tf.cast(segmentation_vis,
                                 dtype=tf.uint8)
      segmentation_summary = tf.py_func(
          utils.general.decode_labels,
          [segmentation_vis, args.num_classes],
          tf.uint8)
      output_vis.append(segmentation_summary)
      # Scalar summary of segmentaiton loss.
      segmentation_loss_summary = tf.summary.scalar(
          'segmentation_loss', segmentation_loss)
      rec_segmentation_loss_summary = tf.summary.scalar(
          'rec_segmentation_loss', rec_segmentation_loss)

    if args.train_depth:
      # Visualize difference.
      depth_vis = tf.image.resize_bilinear(
          tf.abs(outputs[1] - depths),
          tf.shape(image_batch)[1:3,])
      depth_vis /= tf.reduce_max(tf.abs(depth_diff))
      depth_vis = tf.clip_by_value(depth_vis, 0.0, 1.0)
      depth_vis = tf.cast(depth_vis * 255, dtype=tf.uint8)
      depth_summary = tf.tile(depth_vis, [1,1,1,3])
      output_vis.append(depth_summary)
      # Scalar summary of depth loss.
      depth_loss_summary = tf.summary.scalar(
          'depth_loss', depth_loss)
      rec_depth_loss_summary = tf.summary.scalar(
          'rec_depth_loss', rec_depth_loss)

    if args.train_normal:
      # Visualize difference.
      normal_vis = tf.image.resize_bilinear(
          tf.abs(outputs[2] - normals),
          tf.shape(image_batch)[1:3,])
      normal_summary = tf.cast(
          normal_vis / tf.reduce_max(normal_vis) * 255,
          dtype=tf.uint8)
      output_vis.append(normal_summary)
      # Scalar summary of surface normal loss.
      normal_loss_summary = tf.summary.scalar(
          'normal_loss', normal_loss)
      rec_normal_loss_summary = tf.summary.scalar(
          'rec_normal_loss', rec_normal_loss)

    analyzer_max_loss_summary = tf.summary.scalar(
        'analyzer_max_loss', analyzer_max_loss)
    analyzer_min_loss_summary = tf.summary.scalar(
        'analyzer_min_loss', analyzer_min_loss)

    image_summary = tf.summary.image(
        'images', 
        tf.concat(axis=2, values=output_vis),
        max_outputs=args.batch_size)

    total_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(
        args.snapshot_dir,
        graph=tf.get_default_graph())
    
  # Set up tf session and initialize variables. 
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()
    
  sess.run(init)
    
  # Saver for storing checkpoints of the model.
  saver = tf.train.Saver(var_list=tf.global_variables(),
                         max_to_keep=10)
    
  # Load variables if the checkpoint is provided.
  if args.restore_from is not None:
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.restore_from)
    
  # Start queue threads.
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  # Iterate over training steps.
  pbar = tqdm(range(args.num_steps))
  for step in pbar:
    start_time = time.time()
    img_data, lab_data, dph_data, nrm_data= sess.run(datas_batch)
    feed_dict = {step_ph : step,
                 image_batch: img_data,
                 label_batch: lab_data,
                 depth_batch: dph_data,
                 normal_batch: nrm_data}

    sess.run(train_op_a, feed_dict=feed_dict)

    step_loss = 0
    for it in range(args.iter_size):
      # Update summary periodically.
      if it == args.iter_size-1 and step % args.update_tb_every == 0:
        sess_outs = [reduced_loss, total_summary, train_op_s]
        loss_value, summary, _ = sess.run(sess_outs, feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
      else:
        sess_outs = [reduced_loss, train_op_s]
        loss_value, _ = sess.run(sess_outs, feed_dict=feed_dict)

      step_loss += loss_value

    step_loss /= args.iter_size

    lr = sess.run(learning_rate, feed_dict={step_ph: step})

    # Save trained model periodically.
    if step % args.save_pred_every == 0 and step > 0:
      save(saver, sess, args.snapshot_dir, step)

    duration = time.time() - start_time
    desc = 'loss = {:.3f}, lr = {:.6f}'.format(step_loss, lr)
    pbar.set_description(desc)

  coord.request_stop()
  coord.join(threads)
    
if __name__ == '__main__':
  main()
