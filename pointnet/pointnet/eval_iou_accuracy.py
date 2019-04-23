import numpy as np
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--test_area', type=int,help='The areas except this one will be used to estimate the mean instance size')
FLAGS = parser.parse_args()

pred_data_label_file = [line.rstrip() for line in open('./log{}/output_filelist.txt'.format(FLAGS.test_area))]
gt_label_file = [f.rstrip('_pred\.txt') + '_gt.txt' for f in pred_data_label_file]
num_room = len(gt_label_file)
LOG_TEST = open(os.path.join('log{}'.format(FLAGS.test_area), 'log_test.txt'), 'w')

def log_string(out_str):
    LOG_TEST.write(out_str+'\n')
    LOG_TEST.flush()
    print(out_str)

gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]
for i in range(num_room):
    print(i)
    data_label = np.loadtxt(pred_data_label_file[i])
    pred_label = data_label[:,-1]
    gt_label = np.loadtxt(gt_label_file[i])
    print(gt_label.shape)
    for j in xrange(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)


log_string('gt_classes:{}'.format(gt_classes))
log_string('positive_classes:{}'.format(positive_classes))
log_string('true_positive_classes:{}'.format(true_positive_classes))


log_string('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))


log_string('mIoU:')
iou_list = []
for i in range(13):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
    log_string('{}:{}'.format(i,iou))
    iou_list.append(iou)

log_string('mIOU:{}'.format(sum(iou_list)/13.0))