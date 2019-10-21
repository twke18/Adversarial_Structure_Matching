import os
import argparse

from PIL import Image
import numpy as np

from utils.metrics import iou_stats


def parse_argument():
  parser = argparse.ArgumentParser(
      description='Benchmark over 2D-3D-Semantics on segmentation, '\
                   +'depth and surface normals estimation')
  parser.add_argument('--pred_dir', type=str, default='',
                      help='/path/to/prediction.')
  parser.add_argument('--gt_dir', type=str, default='',
                      help='/path/to/ground-truths.')
  parser.add_argument('--depth_unit', type=float, default=512.0,
                      help='Each pixel value difference means 1/depth_unit meters.')
  parser.add_argument('--num_classes', type=int, default=21,
                      help='number of segmentation classes.')
  parser.add_argument('--string_replace', type=str, default=',',
                      help='replace the first string with the second one.')
  parser.add_argument('--train_segmentation', action='store_true',
                      help='enable/disable to benchmark segmentation on mIoU.')
  parser.add_argument('--train_depth', action='store_true',
                      help='enable/disable to benchmark depth.')
  parser.add_argument('--train_normal', action='store_true',
                      help='enable/disable to benchmark surface normal.')
  args = parser.parse_args()

  return args


def benchmark_segmentation(pred_dir, gt_dir, num_classes, string_replace):
  """Benchmark segmentaion on mean Intersection over Union (mIoU).
  """
  print('Benchmarking semantic segmentation.')
  assert(os.path.isdir(pred_dir))
  assert(os.path.isdir(gt_dir))
  tp_fn = np.zeros(num_classes, dtype=np.float64)
  tp_fp = np.zeros(num_classes, dtype=np.float64)
  tp = np.zeros(num_classes, dtype=np.float64)
  for dirpath, dirnames, filenames in os.walk(pred_dir):
    for filename in filenames:
      predname = os.path.join(dirpath, filename)
      gtname = predname.replace(pred_dir, gt_dir)
      if string_replace != '':
        stra, strb = string_replace.split(',')
        gtname = gtname.replace(stra, strb)

      pred = np.asarray(
        Image.open(predname).convert(mode='L'),
        dtype=np.uint8
      )
      gt = np.asarray(
        Image.open(gtname).convert(mode='L'),
        dtype=np.uint8
      )
      _tp_fn, _tp_fp, _tp = iou_stats(
        pred,
        gt,
        num_classes=num_classes,
        background=0
      )
      tp_fn += _tp_fn
      tp_fp += _tp_fp
      tp += _tp

  iou = tp / (tp_fn + tp_fp - tp + 1e-12) * 100.0

  class_names = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'clutter',
                 'column', 'door', 'floor', 'sofa', 'table', 'wall', 'window']

  for i in range(num_classes):
    print('class {:10s}: {:02d}, acc: {:4.4f}%'.format(
      class_names[i], i, iou[i])
    )
  mean_iou = iou.sum() / num_classes
  print('mean IOU: {:4.4f}%'.format(mean_iou))

  mean_pixel_acc = tp.sum() / (tp_fp.sum() + 1e-12)
  print('mean Pixel Acc: {:4.4f}%'.format(mean_pixel_acc))


def benchmark_depth(pred_dir, gt_dir, string_replace):
  """Benchmark depth estimation.
  """
  print('Benchmarking depth estimations.')
  assert(os.path.isdir(pred_dir))
  assert(os.path.isdir(gt_dir))
  N = 0.0
  rmse_linear = 0.0
  rmse_log = 0.0
  absrel = 0.0
  sqrrel = 0.0
  thresholds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  powers = [1/8.0, 1/4.0, 1/2.0, 1.0, 2.0, 3.0]

  for dirpath, dirnames, filenames in os.walk(pred_dir):
    for filename in filenames:
      predname = os.path.join(dirpath, filename)
      gtname = predname.replace(pred_dir, gt_dir)
      if string_replace != '':
        stra, strb = string_replace.split(',')
        gtname = gtname.replace(stra, strb)

      pred = np.asarray(
          Image.open(predname).convert(mode='I'),
          dtype=np.int32)
      gt = np.asarray(
          Image.open(gtname).convert(mode='I'),
          dtype=np.int32)

      pred = np.reshape(pred, (-1,))
      gt = np.reshape(gt, (-1,))
      #mask = np.logical_and(gt >= 51, gt <= 26560)
      mask = gt < 2**16-1

      pred = np.clip(pred, 51, 26560)
      pred = pred[mask].astype(np.float32)/args.depth_unit
      gt = gt[mask].astype(np.float32)/args.depth_unit

      rmse_linear += np.sum((pred-gt)**2)
      rmse_log += np.sum(
          (np.log(np.maximum(pred, 1e-12))-np.log(np.maximum(gt, 1e-12)))**2)
      absrel += np.sum(np.abs(pred-gt)/gt)
      sqrrel += np.sum((pred-gt)**2/gt)
      th = np.maximum(pred/gt, gt/pred)
      for i in range(len(thresholds)):
        #thresholds[i] += np.sum(th < 1.25**(i+1))
        thresholds[i] += np.sum(th < 1.25**powers[i])
      N += pred.shape[0]

  rmse_linear = np.sqrt(rmse_linear/N)
  rmse_log = np.sqrt(rmse_log/N)
  absrel = absrel / N
  sqrrel = sqrrel / N
  for i in range(len(thresholds)):
    thresholds[i] = thresholds[i] / N
  print('RMSE(lin): {:.4f}'.format(rmse_linear))
  print('RMSE(log): {:.4f}'.format(rmse_log))
  print('abs rel: {:.4f}'.format(absrel))
  print('sqr rel: {:.4f}'.format(sqrrel))
  for i in range(len(thresholds)):
    print('\sigma < 1.25^{:.4f}: {:.4f}'.format(powers[i], thresholds[i]))
  print('\sigma < 1.25: {:.4f}'.format(thresholds[0]))
  print('\sigma < 1.25^2: {:.4f}'.format(thresholds[1]))
  print('\sigma < 1.25^3: {:.4f}'.format(thresholds[2]))


def benchmark_normal(pred_dir, gt_dir, string_replace):
  """Benchmark surface normal estimations.
  """
  print('Benchmarking surface normal estimations.')
  assert(os.path.isdir(pred_dir))
  assert(os.path.isdir(gt_dir))
  N = 0.0
  angles = []

  for dirpath, dirnames, filenames in os.walk(pred_dir):
    for filename in filenames:
      predname = os.path.join(dirpath, filename)
      gtname = predname.replace(pred_dir, gt_dir)
      if string_replace != '':
        stra, strb = string_replace.split(',')
        gtname = gtname.replace(stra, strb)

      pred = np.asarray(
          Image.open(predname).convert(mode='RGB'),
          dtype=np.uint8)
      gt = np.asarray(
          Image.open(gtname).convert(mode='RGB'),
          dtype=np.uint8)

      pred = np.reshape(pred, (-1,3))
      gt = np.reshape(gt, (-1,3))
      mask = np.any(gt != 128, axis=-1)

      pred = pred[mask, :].astype(np.float32)-127.5
      gt = gt[mask, :].astype(np.float32)-127.5

      pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True)+1e-12)
      gt = gt / (np.linalg.norm(gt, axis=-1, keepdims=True)+1e-12)

      cos = np.sum(pred*gt, axis=-1)
      abs_cos = np.abs(cos)
      assert(not (abs_cos-1 > 1e-5).any())
      cos = np.clip(cos, -1, 1)
      angles.append(cos)

  angles = np.concatenate(angles, axis=0)
  angles = np.arccos(angles)*(180.0/np.pi)
  print('Angle Mean: {:.4f}'.format(np.mean(angles)))
  print('Angle Median: {:.4f}'.format(np.median(angles)))
  print('Angles within 2.8125: {:.4f}%'.format(np.mean(angles <= 2.8125)*100.0))
  print('Angles within 5.625: {:.4f}%'.format(np.mean(angles <= 5.625)*100.0))
  print('Angles within 11.25: {:.4f}%'.format(np.mean(angles <= 11.25)*100.0))
  print('Angles within 22.5: {:.4f}%'.format(np.mean(angles <= 22.5)*100.0))
  print('Angles within 30: {:.4f}%'.format(np.mean(angles <= 30)*100.0))


if __name__ == '__main__':
  args = parse_argument()
  if args.train_segmentation:
    benchmark_segmentation(args.pred_dir+'/segcls', args.gt_dir+'/segcls',
                           args.num_classes, args.string_replace)
  if args.train_depth:
    benchmark_depth(args.pred_dir+'/depth', args.gt_dir+'/depth',
                    args.string_replace)
  if args.train_normal:
    benchmark_normal(args.pred_dir+'/normal', args.gt_dir+'/normal',
                     args.string_replace)
