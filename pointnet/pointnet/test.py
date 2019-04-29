import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from model import *
import h5py
import indoor3d_util
test_model=6
test_area=6
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--output_filelist',  default='./log{}'.format(test_model),help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--input_test', type=str, default='./data/test_hdf5_file_list_Area{}.txt'.format(test_area), help='Input test data')
parser.add_argument('--restore_model', type=str,default='./log{}'.format(test_model),help='Pretrained model')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', action='store_false',default=False,help='Whether to output OBJ file for prediction visualization.')
FLAGS = parser.parse_args()

TESTING_FILE_LIST = FLAGS.input_test
output_dir=FLAGS.output_filelist
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
ckpt_dir=FLAGS.restore_model

NUM_CLASSES = 13

flog = open(os.path.join(output_dir, 'log_test.txt'), 'w')

def printout(flog, data):
    print(data)
    flog.write(data + '\n')

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def loadDataFile_with_groupseglabel_stanfordindoor(filename):
    f = h5py.File(filename)
    data = f['data'][:]
    if 'label' in f:
        label = f['label'][:].astype(np.int32)
    else :
        label = []
        print ('label ins None')
    return data, label

def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False

test_file_list = getDataFiles(TESTING_FILE_LIST)
test_data = []
test_sem = []
for h5_filename in test_file_list:
    cur_data, cur_sem = loadDataFile_with_groupseglabel_stanfordindoor(h5_filename)
    test_data.append(cur_data)
    test_sem.append(cur_sem)
test_data = np.concatenate(test_data, axis=0)
test_label = np.concatenate(test_sem, axis=0)
test_data = np.asarray(test_data)
test_label = np.asarray(test_label)
print('test_data:', test_data.shape)
print('test_label:', test_label.shape)


def printout(flog, data):
    print(data)
    flog.write(data + '\n')


def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False

def predict():
    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    if not load_checkpoint(ckpt_dir, sess):
        exit()

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}
    gt_classes = [0 for _ in range(NUM_CLASSES)]
    positive_classes = [0 for _ in range(NUM_CLASSES)]
    true_positive_classes = [0 for _ in range(NUM_CLASSES)]
    for batch_idx in range(test_data.shape[0]):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        current_data=test_data[start_idx:end_idx, :, :]
        current_label=test_label[start_idx:end_idx]

        feed_dict = {ops['pointclouds_pl']: current_data,
                     ops['labels_pl']: current_label,
                     ops['is_training_pl']: False}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)
        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:,:,0:12], 2) # BxN
        else:
            pred_label = np.argmax(pred_val, 2) # BxN
        for i in range(pred_label.shape[1]):
            current_label = np.squeeze(current_label)
            pred_label = np.squeeze(pred_label)
            gt_l = int(current_label[i])
            pred_l = int(pred_label[i])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l == pred_l)

    printout(flog, 'gt_l count:{}'.format(gt_classes))
    printout(flog, 'positive_classes count:{}'.format(positive_classes))
    printout(flog, 'true_positive_classes count:{}'.format(true_positive_classes))

    iou_list = []
    for i in range(NUM_CLASSES):
        try:
            iou = true_positive_classes[i] / float(gt_classes[i] + positive_classes[i] - true_positive_classes[i])
            print '{}:{}'.format(i,iou)
        except ZeroDivisionError:
            iou=0
            print '{}:{}'.format(i,iou)
        finally:
            iou_list.append(iou)
    printout(flog, 'IOU:{}'.format(iou_list))
    print 'sum(true_positive_classes):{}'.format(sum(true_positive_classes))
    print 'sum(positive_classes:{}'.format(sum(positive_classes))
    printout(flog, 'ACC:{}'.format(sum(true_positive_classes)*1.0 / (sum(positive_classes))))
    printout(flog, 'mIOU:{}'.format(sum(iou_list) / float(NUM_CLASSES)))


with tf.Graph().as_default():
    predict()
