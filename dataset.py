#!/usr/bin/env python
# coding=utf-8
"""
All image shape is H x W type
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import cv2
import os

class Config(object): 
    def __init__(self): 
        self.image_shape = None 
        self.output_shape = None 
        self.num_joints = None

config = Config()
config.image_shape = (256, 256)  # H x W
config.output_shape = (64, 64)  # H x W
config.output_depth = 64
config.num_joints = 14


def get_img(dataset_dir, index):
    img_dir = dataset_dir + 'images/'
    img_path = img_dir + str(index).zfill(5) + '.jpg'
    img = plt.imread(img_path)
    return img


def load_index_label(dataset_dir, index):
    """ load index-th label from saved pickle file
    @Returns:
        joints: (3, N_nodes)
        ordinal: (N_nodes, N_nodes)
    """
    pkl_list = pickle.load(open(dataset_dir + 'label_mpii_lsp.pkl', 'rb'))
    pkl = pkl_list[index]
    ind = pkl['index']
    assert ind == str(index).zfill(5)
    joints = pkl['joints']
    ordinal = pkl['ordinal']
    return joints, ordinal


def get_label(pkl_list, index):
    """ load index-th label from list of dict
    @Returns:
        joints: (3, N_nodes)
        ordinal: (N_nodes, N_nodes)
    """
    pkl = pkl_list[index]
    ind = pkl['index']
    assert ind == str(index).zfill(5)
    joints = pkl['joints']
    ordinal = pkl['ordinal']
    return joints, ordinal


def warpAffine_sample(img, joints, image_shape=config.image_shape, dr=80):
    """ crop and warp image
    @Args:
        img: 3-channel image
        joints: (3, N_nodes)
        config.image_shape: parameters from config, expected training image shape
    """
    min_jx = np.min(joints[0, :])
    max_jx = np.max(joints[0, :])
    min_jy = np.min(joints[1, :])
    max_jy = np.max(joints[1, :])
    c_x = (min_jx + max_jx) / 2.0
    c_y = (min_jy + max_jy) / 2.0
    
    r = np.maximum(max_jx - min_jx, max_jy - min_jy) / 2.0
    r = int(r) + dr

    min_x = int(c_x) - r
    max_x = int(c_x) + r
    min_y = int(c_y) - r
    max_y = int(c_y) + r

    pts1 = np.float32([[min_x, min_y], [max_x, min_y], [min_x, max_y]])
    pts2 = np.float32([[0, 0], [image_shape[1], 0], [0, image_shape[0]]]) # ! shape[0]:h
    mat = cv2.getAffineTransform(pts1, pts2)
    img_warped  = cv2.warpAffine(img, mat, (image_shape[1], image_shape[0]))
    # !!! cv2.warpAffine : 3rd para: (row, col), but img_warped.shape = (col, row, 3)

    joints_warped = joints.copy()
    pts = joints.copy()
    pts[-1, :] = 1
    pts = np.dot(mat, pts).astype(np.int32)
    joints_warped[:2, :] = pts
    return img_warped, joints_warped


def gen_joints_heatmap(joints, ori_size=config.image_shape, tar_size=config.output_shape, num_joints=config.num_joints):
    """ generate GT heatmap label for joints
    @Args:
        joints: (3, num_joints), x/y/visible, x/y is in ori_size coordinate
        ori_size: (H, W)
        tar_size: (H, W)
        num_joints: int
    @Returns:
        ret: gaussian blured heatmap (num_joints, tar_size[0], tar_size[1])
    """
    ret = np.zeros( (num_joints, tar_size[0], tar_size[1]), dtype='float32' )  # D x H x W
    # scale joints x/y to [0, 1]
    label_x = joints[0] / ori_size[1]
    label_y = joints[1] / ori_size[0]
    for j in range(num_joints) :
        if label_x[j] < 0 or label_y[j] < 0 or label_x[j] > 0.999 or label_y[j] > 0.999:
            continue
        ret[j][ int(label_y[j] * tar_size[0]) ][ int(label_x[j] * tar_size[1]) ] = 1
    ret = np.transpose( ret, (1, 2, 0) )
    ret = cv2.GaussianBlur( ret, (7, 7), 0 )  # the image can have any number of channels
    ret = np.transpose( ret, (2, 0, 1) )
    for j in range(num_joints) :
        am = np.amax( ret[j] )
        if am == 0 :
            continue
        ret[j] /= am
    return ret
 

def visuaize_skeleton(joints):
    """
    @Args:
        joints: 2 x 14
    """
    r_skeleton = [[0, 1], [1, 2], [6, 7], [7, 8]]
    l_skeleton = [[3, 4], [4, 5], [9, 10], [10, 11]]
    for [i, j] in r_skeleton:
        plt.plot([joints[0, i], joints[0, j]], [joints[1, i], joints[1, j]], 'g')
    for [i, j] in l_skeleton:
        plt.plot([joints[0, i], joints[0, j]], [joints[1, i], joints[1, j]], 'b')
    plt.plot([joints[0, 12], joints[0, 13]], [joints[1, 12], joints[1, 13]], 'y')
    pelvis = (joints[:, 2] + joints[:, 3]) / 2.0
    plt.plot([joints[0, 12], pelvis[0]], [joints[1, 12], pelvis[1]], 'y')


def visualize_index(dataset_dir, index):
    """ visualize index-th image
    @Args:
        dataset_dir: directory to save images
        index: int
    """
    joints, ordinal = load_index_label(dataset_dir, index)
    img_dir = dataset_dir + 'images/'
    img_path = img_dir + str(index).zfill(5) + '.jpg'
    img = plt.imread(img_path)
    plt.figure()
    plt.imshow(img)
    joints_inst = joints[:2, :]
    joints_visb = joints[2, :]
    ordinal_inst = ordinal[:, :]
    # attention: MPII & LSP: visible definition diff, here visble = 1
    for j in range(len(joints_visb)):
        if joints_visb[j] == 1:
            plt.scatter(joints_inst[0, j], joints_inst[1, j], marker='.', c='r', s=100)
    visuaize_skeleton(joints_inst)
    plt.title(index)
    plt.show()


def visualize_sample(img, joints, auto_close=False):
    """ visualize image and joints pair
    @Args:
        img: (H, W, 3)
        joints: (3, N_nodes)
    """
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    joints_inst = joints[:2, :]
    joints_visb = joints[2, :]
    # attention: MPII & LSP: visible definition diff, here visble = 1
    # for j in range(len(joints_visb)):
    #     if joints_visb[j] == 1:
    #         plt.scatter(joints_inst[0, j], joints_inst[1, j], marker='.', c='r', s=100)
    plt.scatter(joints_inst[0, :], joints_inst[1, :], marker='.', c='r', s=100)
    visuaize_skeleton(joints_inst)

    plt.subplot(122)  # plot heatmap
    heatmap = gen_joints_heatmap(joints)
    heatmap = np.sum(heatmap, axis=0)
    plt.imshow(heatmap)

    if auto_close:
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    else:
        plt.show()
   

class DataLoader(object):
    def __init__(self, dataset_dir=None, status='train' ,batch_size=32):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.status = status

        self.img_names = None
        self.img_num = None
        self.label_list = None
        self.dataset = None
        self.iter = None
        self._build(status)

    def _data_process_pyfunc(self, img_name, index):
        # !!! in py_func, don't use tf function!
        # img = tf.read_file(img_name)
        # img_jpg = tf.image.decode_jpeg(img, channels=3)
        img = cv2.imread(img_name.decode('utf-8'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.label_list[index]
        index = label['index']
        joints = label['joints']
        ordinal = label['ordinal']

        img, joints = warpAffine_sample(img, joints)
        img = img / 255.0
        heatmap = gen_joints_heatmap(joints)
        return np.int16(index), img.astype(np.float32), joints.astype(np.float32),\
                heatmap.astype(np.float32), ordinal.astype(np.float32)

    # train : return mini-batch
    # test  : return single component
    def _build(self, status=None):
        # build dataset
        img_names = os.listdir(self.dataset_dir + 'images/')
        img_names.sort()  # !!! sort
        self.img_names = [self.dataset_dir + 'images/' + fil for fil in img_names]
        self.img_num = len(self.img_names)
        self.label_list = pickle.load(open(self.dataset_dir + 'label_mpii_lsp.pkl', 'rb'))
        label_index_list = list(range(self.img_num))

        dataset = tf.data.Dataset.from_tensor_slices((self.img_names, label_index_list))
        dataset = dataset.shuffle(buffer_size=self.img_num) # shuffle before heavy transformation !
        dataset = dataset.map(lambda img_name, index : tuple(tf.py_func(self._data_process_pyfunc, \
                [img_name, index], [tf.int16, tf.float32, tf.float32, tf.float32, tf.float32])))

        # judge status
        if status == 'train':
            self.dataset = dataset.repeat().batch(self.batch_size)      # first repeat then batch !!
        elif status == 'test':
            self.dataset = dataset.repeat()
        else:
            raise AttributeError("status must be 'train' or 'test' !")
        # iter
        self.iter = self.dataset.make_one_shot_iterator()
       
    # return next_batch_op
    def load_batch(self):
        """
        @Returns:
            index.shape = (batch_size,)
            img.shape = (batch_size, image_shape, image_shape, 3)
            joints.shape = (batch_size, 3, num_joints) 
            heatmap.shape = (batch_size, num_joints, output_shape, output_shape)
            ordinal.shape = (batch_size, num_joints, num_joints)
        """
        return self.iter.get_next()



if __name__ == '__main__':
    dataset_dir = './dataset/'
    dataLoader = DataLoader(dataset_dir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # train_next_batch_op = dataLoader.load_batch()
        # index, img, joints, heatmap, ordinal = sess.run(train_next_batch_op)

        index_op, img_op, joints_op, heatmap_op, ordinal_op = dataLoader.load_batch()
        index, img, joints, heatmap, ordinal = sess.run([index_op, img_op, joints_op, heatmap_op, ordinal_op])
        for i in range(img.shape[0]):
            print(index[i])
            visualize_sample(img[i], joints[i]) #, auto_close=True)

        from IPython import embed; embed()

        
    """
    # for i in list(range(10)) + list(range(13029, 13035)):
    #     visual_index(dataset_dir, i)

    pkl_list = pickle.load(open(dataset_dir + 'label_mpii_lsp.pkl', 'rb'))
    for i in list(range(10)) + list(range(13029, 13035)):
        img =  get_img(dataset_dir, i)
        joints, _ = get_label(pkl_list, i)
        # visualize_sample(img, joints)
        img, joints = warpAffine_sample(img, joints)
        heatmap = gen_joints_heatmap(joints)
        visualize_sample(img, joints) #, auto_close=True)
        # from IPython import embed; embed()
    """

    
    

