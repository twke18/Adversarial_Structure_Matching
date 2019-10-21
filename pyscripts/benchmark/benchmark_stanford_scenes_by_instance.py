import os
import argparse

from PIL import Image
import numpy as np
import json

from utils.metrics import iou_stats


def parse_argument():
  parser = argparse.ArgumentParser(
      description='Benchmark over 2D-3D-Semantics on segmentation, '\
                   +'depth and surface normals estimation')
  parser.add_argument('--pred_dir', type=str, default='',
                      help='/path/to/prediction.')
  parser.add_argument('--gt_dir', type=str, default='',
                      help='/path/to/ground-truths.')
  parser.add_argument('--inst_class_json', type=str, default='',
                      help='/path/to/json/of/inst/classes.')
  parser.add_argument('--depth_unit', type=float, default=512.0,
                      help='Each pixel value difference means 1/depth_unit '
                           'meters.')
  parser.add_argument('--num_classes', type=int, default=21,
                      help='number of segmentation classes.')
  parser.add_argument('--string_replace', type=str, default=',',
                      help='replace the first string with the second one.')
  parser.add_argument('--train_depth', action='store_true',
                      help='enable/disable to benchmark depth.')
  parser.add_argument('--train_normal', action='store_true',
                      help='enable/disable to benchmark surface normal.')
  args = parser.parse_args()

  return args


def benchmark_depth(pred_dir, gt_dir, inst_dir,
                    instance_labels, string_replace):
  """Benchmark depth estimation.
  """
  print('Benchmarking instance-level depth estimations.')
  assert(os.path.isdir(pred_dir))
  assert(os.path.isdir(gt_dir))
  assert(os.path.isdir(inst_dir))
  num_instance = len(instance_labels)
  num_pixels = np.zeros(num_instance, dtype=np.float64)
  rmse_linear = np.zeros(num_instance, dtype=np.float64)
  rmse_log = np.zeros(num_instance, dtype=np.float64)
  absrel = np.zeros(num_instance, dtype=np.float64)
  sqrrel = np.zeros(num_instance, dtype=np.float64)
  thresholds = [np.zeros(num_instance, dtype=np.float64) for _ in range(5)]

  for dirpath, dirnames, filenames in os.walk(pred_dir):
    for filename in filenames:
      predname = os.path.join(dirpath, filename)
      gtname = predname.replace(pred_dir, gt_dir)
      instname = gtname.replace('depth', 'semantic')
      if string_replace != '':
        stra, strb = string_replace.split(',')
        gtname = gtname.replace(stra, strb)

      pred = np.asarray(
          Image.open(predname).convert(mode='I'),
          dtype=np.int32)
      gt = np.asarray(
          Image.open(gtname).convert(mode='I'),
          dtype=np.int32)
      inst = np.asarray(
          Image.open(instname).convert(mode='RGB'),
          dtype=np.uint8)
      inst = inst[:,:,0]*256**2+inst[:,:,1]*256+inst[:,:,2]

      pred = np.reshape(pred, (-1,))
      gt = np.reshape(gt, (-1,))
      inst = np.reshape(inst, (-1,))
      mask = gt < 2**16-1

      pred = np.clip(pred, 51, 26560)
      pred = pred[mask].astype(np.float32)/args.depth_unit
      gt = gt[mask].astype(np.float32)/args.depth_unit
      inst = inst[mask]

      for inst_ind in np.unique(inst):
        if inst_ind == 855309:
          continue
        m = inst == inst_ind
        if not m.any():
          continue

        pred_m = pred[m]
        gt_m = gt[m]
        rmse_linear[inst_ind] += np.sum((pred_m-gt_m)**2)
        rmse_log[inst_ind] += np.sum(
            (np.log(pred_m)-np.log(gt_m))**2)
        absrel[inst_ind] += np.sum(np.abs(pred_m-gt_m)/gt_m)
        sqrrel[inst_ind] += np.sum((pred_m-gt_m)**2/gt_m)
        th = np.maximum(pred_m/gt_m, gt_m/pred_m)
        for i in range(len(thresholds)):
          thresholds[i][inst_ind] += np.sum(th < 1.25**(np.power(2.0, i-2)))
        num_pixels[inst_ind] += m.sum()

  # instance level metrics.
  num_pixels = np.maximum(num_pixels, 1e-12)
  rmse_linear = np.sqrt(rmse_linear/num_pixels)
  rmse_log = np.sqrt(rmse_log/num_pixels)
  absrel = absrel / num_pixels
  sqrrel = sqrrel / num_pixels
  for i in range(len(thresholds)):
    thresholds[i] = thresholds[i] / num_pixels

  # semantic level metrics.
  cls_names = {}
  cls_num_insts = []
  cls_rmse_linear = []
  cls_rmse_log = []
  cls_absrel = []
  cls_sqrrel = []
  cls_thresholds = [[] for t in thresholds]
  for inst_ind, inst_name in enumerate(instance_labels):
    cls_name = inst_name.split('_')[0]
    if cls_names.get(cls_name, None) is None:
      cls_names[cls_name] = len(cls_names)
      cls_rmse_linear.append(0.0)
      cls_rmse_log.append(0.0)
      cls_absrel.append(0.0)
      cls_sqrrel.append(0.0)
      for t in cls_thresholds:
        t.append(0.0)
      cls_num_insts.append(0)

    if num_pixels[inst_ind] >= 1:
      cls_ind = cls_names[cls_name]
      cls_num_insts[cls_ind] += 1
      cls_rmse_linear[cls_ind] += rmse_linear[inst_ind]
      cls_rmse_log[cls_ind] += rmse_log[inst_ind]
      cls_absrel[cls_ind] += absrel[inst_ind]
      cls_sqrrel[cls_ind] += sqrrel[inst_ind]
      for ct, it in zip(cls_thresholds, thresholds):
        ct[cls_ind] += it[inst_ind]
  
  cls_num_insts = np.maximum(np.array(cls_num_insts), 1e-12)
  cls_rmse_linear = np.array(cls_rmse_linear) / cls_num_insts
  cls_rmse_log = np.array(cls_rmse_log) / cls_num_insts
  cls_absrel = np.array(cls_absrel) / cls_num_insts
  cls_sqrrel = np.array(cls_sqrrel) / cls_num_insts
  for i in range(len(cls_thresholds)):
    cls_thresholds[i] = np.array(cls_thresholds[i]) / cls_num_insts

  for cls_name, cls_ind in cls_names.items():
    print('class {:s}, RMSE(lin): {:.4f}'.format(
        cls_name, cls_rmse_linear[cls_ind]))
    print('class {:s}, RMSE(log): {:.4f}'.format(
        cls_name, cls_rmse_log[cls_ind]))
    print('class {:s}, abs rel: {:.4f}'.format(
        cls_name, cls_absrel[cls_ind]))
    print('class {:s}, sqr rel: {:.4f}'.format(
        cls_name, cls_sqrrel[cls_ind]))
    for i in range(len(cls_thresholds)):
      print('class {:s}, \sigma < 1.25^{:.4f}: {:.4f}'.format(
          cls_name, np.power(2.0, i-2), cls_thresholds[i][cls_ind]))
    print('class {:s}, \sigma < 1.25: {:.4f}'.format(
        cls_name, cls_thresholds[0][cls_ind]))
    print('class {:s}, \sigma < 1.25^2: {:.4f}'.format(
        cls_name, cls_thresholds[1][cls_ind]))
    print('class {:s}, \sigma < 1.25^3: {:.4f}'.format(
        cls_name, cls_thresholds[2][cls_ind]))


def benchmark_normal(pred_dir, gt_dir, inst_dir,
                     instance_labels, string_replace):
  """Benchmark surface normal estimations.
  """
  print('Benchmarking instance-level surface normal estimations.')
  assert(os.path.isdir(pred_dir))
  assert(os.path.isdir(gt_dir))
  assert(os.path.isdir(inst_dir))
  num_instance = len(instance_labels)
  angles = [[] for _ in range(num_instance)]

  for dirpath, dirnames, filenames in os.walk(pred_dir):
    for filename in filenames:
      predname = os.path.join(dirpath, filename)
      gtname = predname.replace(pred_dir, gt_dir)
      instname = (gtname.replace('normal', 'semantic')
                        .replace('semantics', 'semantic'))
      if string_replace != '':
        stra, strb = string_replace.split(',')
        gtname = gtname.replace(stra, strb)

      pred = np.asarray(
          Image.open(predname).convert(mode='RGB'),
          dtype=np.uint8)
      gt = np.asarray(
          Image.open(gtname).convert(mode='RGB'),
          dtype=np.uint8)
      inst = np.asarray(
          Image.open(instname).convert(mode='RGB'),
          dtype=np.uint8)
      inst = inst[:,:,0]*256**2+inst[:,:,1]*256+inst[:,:,2]

      pred = np.reshape(pred, (-1,3))
      gt = np.reshape(gt, (-1,3))
      inst = np.reshape(inst, (-1,))
      mask = np.any(gt != 128, axis=-1)

      pred = pred[mask, :].astype(np.float32)-127.5
      gt = gt[mask, :].astype(np.float32)-127.5
      inst = inst[mask]

      pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True)+1e-12)
      gt = gt / (np.linalg.norm(gt, axis=-1, keepdims=True)+1e-12)

      cos = np.sum(pred*gt, axis=-1)
      abs_cos = np.abs(cos)
      assert(not (abs_cos-1 > 1e-5).any())
      cos = np.clip(cos, -1, 1)

      for inst_ind in np.unique(inst):
        if inst_ind == 855309:
          continue
        m = inst == inst_ind
        if m.any():
          angles[inst_ind].append(cos[m])

  # semantic level metrics.
  cls_names = {}
  cls_mean_angles = []
  cls_med_angles = []
  cls_angles_3 = []
  cls_angles_6 = []
  cls_angles_11 = []
  cls_angles_22 = []
  cls_angles_30 = []
  cls_num_insts = []
  for inst_ind, inst_name in enumerate(instance_labels):
    cls_name = inst_name.split('_')[0]
    if cls_names.get(cls_name, None) is None:
      cls_names[cls_name] = len(cls_names)
      cls_mean_angles.append(0.0)
      cls_med_angles.append(0.0)
      cls_angles_3.append(0.0)
      cls_angles_6.append(0.0)
      cls_angles_11.append(0.0)
      cls_angles_22.append(0.0)
      cls_angles_30.append(0.0)
      cls_num_insts.append(0)

    inst_angs = angles[inst_ind]
    if len(inst_angs) > 0:
      inst_angs = np.hstack(inst_angs)
      inst_angs = np.arccos(inst_angs)*(180.0/np.pi)
      cls_ind = cls_names[cls_name]

      cls_mean_angles[cls_ind] += np.mean(inst_angs)
      cls_med_angles[cls_ind] += np.median(inst_angs)
      cls_angles_3[cls_ind] += np.mean(inst_angs <= 2.8125)
      cls_angles_6[cls_ind] += np.mean(inst_angs <= 5.625)
      cls_angles_11[cls_ind] += np.mean(inst_angs <= 11.25)
      cls_angles_22[cls_ind] += np.mean(inst_angs <= 22.5)
      cls_angles_30[cls_ind] += np.mean(inst_angs <= 30)
      cls_num_insts[cls_ind] += 1

  cls_num_insts = np.maximum(np.array(cls_num_insts), 1e-12)
  cls_mean_angles = np.array(cls_mean_angles) / cls_num_insts
  cls_med_angles = np.array(cls_med_angles) / cls_num_insts
  cls_angles_3 = np.array(cls_angles_3) / cls_num_insts
  cls_angles_6 = np.array(cls_angles_6) / cls_num_insts
  cls_angles_11 = np.array(cls_angles_11) / cls_num_insts
  cls_angles_22 = np.array(cls_angles_22) / cls_num_insts
  cls_angles_30 = np.array(cls_angles_30) / cls_num_insts

  for cls_name, cls_ind in cls_names.items():
    print('class {:s}, Angle Mean: {:.4f}'.format(
        cls_name, cls_mean_angles[cls_ind]))
    print('class {:s}, Angle Median: {:.4f}'.format(
        cls_name, cls_med_angles[cls_ind]))
    print('class {:s}, Angle within 2.8125: {:.4f}'.format(
        cls_name, cls_angles_3[cls_ind] * 100.0))
    print('class {:s}, Angle within 5.625: {:.4f}'.format(
        cls_name, cls_angles_6[cls_ind] * 100.0))
    print('class {:s}, Angle within 11.25: {:.4f}'.format(
        cls_name, cls_angles_11[cls_ind] * 100.0))
    print('class {:s}, Angle within 22.5: {:.4f}'.format(
        cls_name, cls_angles_22[cls_ind] * 100.0))
    print('class {:s}, Angle within 30: {:.4f}'.format(
        cls_name, cls_angles_30[cls_ind] * 100.0))


if __name__ == '__main__':
  args = parse_argument()

  with open(args.inst_class_json) as json_file:
    instance_labels = json.load(json_file)

  if args.train_depth:
    benchmark_depth(
        args.pred_dir+'/depth', args.gt_dir+'/depth', args.gt_dir+'/semantic',
        instance_labels, args.string_replace)
  if args.train_normal:
    benchmark_normal(
        args.pred_dir+'/normal', args.gt_dir+'/normal', args.gt_dir+'/semantic',
        instance_labels, args.string_replace)
