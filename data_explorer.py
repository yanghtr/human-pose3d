#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pickle
import os

def mat2npy_label(dataset_dir):
    """ convert .mat label file (lsp-mpii-ordinal) to numpy array
    @Args:
        dataset_dir: dataset directory
    @Returns:
        joints: (3, N_nodes, N_sample)
        ordinal: (N_sample, N_nodes, N_nodes)
    """
    joints_mat = scipy.io.loadmat(dataset_dir + 'joints.mat')
    ordinal_mat = scipy.io.loadmat(dataset_dir + 'ordinal.mat')
    joints = joints_mat['joints']
    ordinal = ordinal_mat['ord']
    return joints, ordinal


def reformat_mpii_label(dataset_dir):
    """ reformat dataset (mpii), convert to 14 nodes and reorder
    """
    joints_mpii, ordinal_mpii = mat2npy_label(dataset_dir)
    joints_mpii_reorder = np.delete(joints_mpii, [6, 7], axis=1)
    ordinal_mpii_reorder = np.delete(ordinal_mpii, [6, 7], axis=1)
    ordinal_mpii_reorder = np.delete(ordinal_mpii_reorder, [6, 7], axis=2)

    reorderInd = np.array(list(range(6)) + list(range(8, 14)) + [6, 7])
    joints_mpii_reorder = joints_mpii_reorder[:, reorderInd, :]
    ordinal_mpii_reorder = ordinal_mpii_reorder[:, :, reorderInd]
    ordinal_mpii_reorder = ordinal_mpii_reorder[:, reorderInd, :]
    # from IPython import embed; embed()
    return joints_mpii_reorder, ordinal_mpii_reorder


def dump_label(dataset_dir, joints, ordinal):
    """dump label (lsp-mpii-ordinal) to pickle
    @Args:
        dataset_dir: directory where to dump lable.pkl
        joints: (3, N_nodes, N_sample)
        ordinal: (N_sample, N_nodes, N_nodes)
    """
    dlist = []
    assert joints.shape[2] == ordinal.shape[0]
    for i in range(joints.shape[2]):
        dtmp = {}
        dtmp['index'] = str(i).zfill(5)
        dtmp['joints'] = joints[:, :, i]
        dtmp['ordinal'] = ordinal[i]
        dlist.append(dtmp)

    with open(dataset_dir + 'label.pkl', 'wb') as f:
        pickle.dump(dlist, f, protocol=2)


def load_index_label(dataset_dir, index):
    """ load index-th label from saved pickle file
    @Returns:
        joints: (3, N_nodes)
        ordinal: (N_nodes, N_nodes)
    """
    pkl_list = pickle.load(open(dataset_dir + 'label.pkl', 'rb'))
    pkl = pkl_list[index]
    ind = pkl['index']
    assert ind == str(index).zfill(5)
    joints = pkl['joints']
    ordinal = pkl['ordinal']
    return joints, ordinal


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


def visual(dataset_dir, joints, ordinal):
    # joints, ordinal = mat2npy(dataset_dir)
    img_dir = dataset_dir + 'images/'
    file_list = os.listdir(img_dir)
    file_list.sort()  # remember to sort !
    N_sample = len(file_list)
    for i, img_name in enumerate(file_list):
        img_path = img_dir + img_name
        img = plt.imread(img_path)
        plt.figure()
        plt.imshow(img)
        joints_inst = joints[:2, :, i]
        joints_visb = joints[2, :, i]
        ordinal_inst = ordinal[i, :, :]
        for j in range(len(joints_visb)):
            if joints_visb[j] == 0:
                plt.scatter(joints_inst[0, j], joints_inst[1, j], marker='.', s=100)
        plt.scatter(joints_inst[0, :], joints_inst[1, :], marker='.', c='r', s=100)
        visuaize_skeleton(joints_inst)
        plt.title(i)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


if __name__ == '__main__':
    # convert and dump mpii label
    joints_mpii, ordinal_mpii = reformat_mpii_label('./mpii_upis1h/')
    # To visualize, uncomment visual(...)
    # visual('./mpii_upis1h/', joints_mpii, ordinal_mpii)
    dump_label('./mpii_upis1h/', joints_mpii, ordinal_mpii)

    # convert, reorder and dump lsp label
    joints_lsp, ordinal_lsp = mat2npy_label('./lsp_dataset_original/')
    # visual('./lsp_dataset_original/', joints_lsp, ordinal_lsp)
    dump_label('./lsp_dataset_original/', joints_lsp, ordinal_lsp)

    """ directory tree
        ./lsp-mpii-ordinal
        ├── data_explorer.py
        ├── lsp_dataset_original
        │   ├── images
        │   ├── joints.mat
        │   ├── label.pkl
        │   ├── LICENSE
        │   ├── ordinal.mat
        │   └── README.txt
        ├── mpii_upis1h
        │   ├── images
        │   ├── joints.mat
        │   ├── label.pkl
        │   ├── LICENSE
        │   └── ordinal.mat
        └── README
    """

