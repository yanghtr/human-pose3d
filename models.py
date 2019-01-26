#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn

from dataset import *

def residual(x, channel_in, channel_out, name, trainable=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        xmain = tflearn.layers.normalization.batch_normalization(x, trainable=trainable, name='bn0')
        xmain = tflearn.activations.relu(xmain)
        xmain = tflearn.layers.conv.conv_2d(xmain, int(channel_out/2), [1,1], strides=1, activation='relu',\
                        weight_decay=1e-5, regularizer='L2', trainable=trainable, name='conv0')

        xmain = tflearn.layers.normalization.batch_normalization(xmain, trainable=trainable, name='bn1')
        xmain = tflearn.activations.relu(xmain)
        xmain = tflearn.layers.conv.conv_2d(xmain, int(channel_out/2), [3,3], strides=1, activation='relu',\
                        weight_decay=1e-5, regularizer='L2', trainable=trainable, name='conv1')

        xmain = tflearn.layers.normalization.batch_normalization(xmain, trainable=trainable, name='bn2')
        xmain = tflearn.activations.relu(xmain)
        xmain = tflearn.layers.conv.conv_2d(xmain, channel_out, [1,1], strides=1, activation='relu',\
                        weight_decay=1e-5, regularizer='L2', trainable=trainable, name='conv2')

        xbranch = tflearn.layers.conv.conv_2d(x, channel_out, [1,1], strides=1, activation='relu',\
                        weight_decay=1e-5, regularizer='L2', trainable=trainable, name='conv3')

        xout = tf.add(xmain, xbranch)

        return xout


def hourGlass(x, channel_in, channel_out, name, trainable=True, reuse=False):
    """ 4-order hourGlass
    """
    with tf.variable_scope(name, reuse=reuse):
        xmain = x
        xmid_down_list = []
        xbranch_list = []
        for i in range(4):
            xmid_down_list.append(xmain)
            xmain = tflearn.layers.conv.max_pool_2d(xmain, [2,2], strides=2, name='mpool'+str(i)) 
            for j in range(3):
                xmain = residual(xmain, channel_in, channel_out, trainable=trainable, reuse=reuse, name='residual_'+str(i)+'/'+str(j))

        for i in range(4):
            xbranch = xmid_down_list[i]
            for j in range(3):
                xbranch = residual(xbranch, channel_in, channel_out, trainable=trainable, reuse=reuse, name='residual_'+str(i+4)+'/'+str(j))
            xbranch_list.append(xbranch)

        xmain = residual(xmain, channel_in, channel_out, trainable=trainable, reuse=reuse, name='residual_8')
        for i in range(4):
            xmain = residual(xmain, channel_in, channel_out, trainable=trainable, reuse=reuse, name='residual_'+str(i+9))
            xmain = tflearn.layers.conv.upsample_2d(xmain, [2,2], name='upool'+str(i))
            xmain = tf.add(xmain, xbranch_list[3-i])

    return xmain


class VolumeHumanPose(object):
    def __init__(self, dataset_next_batch, learning_rate, trainable=True, reuse=False):
        self.dataset_next_batch = dataset_next_batch
        self.trainable = trainable
        self.reuse = reuse

        self.loss = 0
        self._loss()

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        self.opt_op = self.optimizer.minimize(self.loss)


    def inference(self, img_batch):
        with tf.variable_scope('VolumeHumanPose', reuse=self.reuse):
            x0 = tflearn.layers.conv.conv_2d(img_batch, 64, [7,7], strides=2, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv0')
            x0 = tflearn.layers.conv.conv_2d(x0, 128, [3,3], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv1')
            x0 = tflearn.layers.conv.max_pool_2d(x0, [2,2], strides=2, name='mpool0') 

            x1 = hourGlass(x0, 128, 256, trainable=self.trainable, reuse=self.reuse, name='hourGlass0')
            x1 = tflearn.layers.conv.conv_2d(x1, 512, [1,1], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv2')
            x1 = tflearn.layers.conv.conv_2d(x1, 256, [1,1], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv3')

            # heatmap:
            heatmap_mid = tflearn.layers.conv.conv_2d(x1, config.num_joints, [1,1], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv4')
            x2 = tflearn.layers.conv.conv_2d(heatmap_mid, 384, [1,1], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv5')
            self.heatmap_mid = tf.transpose(heatmap_mid, [0, 3, 1, 2])

            x3 = tf.concat([x0, x1], axis=3)
            assert x3.shape[-1] == 384
            x3 = tflearn.layers.conv.conv_2d(x3, 384, [3,3], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv6')

            x4 = tf.add(x2, x3)
            x4 = tflearn.layers.conv.conv_2d(x4, 384, [3,3], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv7')

            x5 = hourGlass(x4, 384, 256, trainable=self.trainable, reuse=self.reuse, name='hourGlass1')
            x5 = tflearn.layers.conv.conv_2d(x5, 512, [1,1], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv8')
            x5 = tflearn.layers.conv.conv_2d(x5, 512, [1,1], strides=1, activation='relu',\
                            weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv9')
            self.outVoxel = tflearn.layers.conv.conv_2d(x5, config.output_depth * config.num_joints, [1,1], strides=1,\
                            activation='relu', weight_decay=1e-5, regularizer='L2', trainable=self.trainable, name='conv10')

            return self.heatmap_mid, self.outVoxel


    def _ranking_loss(self, z_batch, ordinal_batch):
        """ ranking loss
        @Args:
            z_batch: (batch_size, num_joints)
            ordinal_batch: (batch_size, num_joints, num_joints)
        """
        def _genMatrixH(v): 
            """ vector batch to matrix batch
            @Args:
                v: (batch_size, *num_joints)
            @Returns:
                m: (batch_size, num_joints(new tile dim), *num_joints)
            """
            m = tf.tile(v, [1, v.shape[1]]) 
            m = tf.reshape(m, [v.shape[0], v.shape[1], v.shape[1]]) 
            return m 

        z_mat_j = _genMatrixH(z_batch)
        z_mat_i = tf.transpose(z_mat_j, [0, 2, 1])
        z_mat_ij = z_mat_i - z_mat_j
        ranking_loss_mat1 = ( tf.log(tf.ones_like(z_mat_ij) + tf.exp(tf.multiply(z_mat_ij, ordinal_batch))) ) / 2.0
        ranking_loss = tf.reduce_sum(ranking_loss_mat1)
        ranking_loss_mat2 = tf.multiply( (tf.ones_like(ordinal_batch) - tf.abs(ordinal_batch)), tf.multiply(z_mat_ij, z_mat_ij) )
        ranking_loss += tf.reduce_sum(ranking_loss_mat2)
        
        return ranking_loss


    def _loss(self):
        self.index_batch, self.img_batch, self.joints_batch, self.heatmap_batch, \
                self.ordinal_batch = self.dataset_next_batch

        heatmap_mid_pred_batch, outVoxel_pred_batch = self.inference(self.img_batch) 

        # convert volume to heatmap and z
        z_range = tf.constant(np.array(list(range(config.output_depth)), np.float32)[:, np.newaxis])
        heatmap_voxel_pred_batch_list = []
        z_voxel_pred_batch_list = []
        for i in range(config.num_joints):
            voxel = outVoxel_pred_batch[:, :, :, config.output_depth * i : config.output_depth * (i+1)]
            # voxel : (batch_size, 64, 64, 64)
            pxy = tf.reduce_sum(voxel, axis=3)
            pz = tflearn.layers.conv.avg_pool_2d(voxel, config.output_shape) * \
                    (config.output_shape[0] * config.output_shape[1])
            pz = tf.squeeze(pz)
            z = tf.matmul(pz, z_range)

            heatmap_voxel_pred_batch_list.append(pxy)  # pxy: (batch_size, 64, 64)
            z_voxel_pred_batch_list.append(z)  # z: (batch_size, 1)
            
        heatmap_voxel_pred_batch = tf.stack(heatmap_voxel_pred_batch_list, axis=1)  # pxy: (batch_size, num_joints, 64, 64)
        z_voxel_pred_batch = tf.stack(z_voxel_pred_batch_list, axis=1)  # z: (batch_size, num_joints, 1)
        z_voxel_pred_batch = tf.squeeze(z_voxel_pred_batch)  # z: (batch_size, num_joints)

        # loss function
        self.heatmap_mid_loss = tf.losses.mean_squared_error(labels=self.heatmap_batch, predictions=heatmap_mid_pred_batch)
        self.heatmap_voxel_loss = tf.losses.mean_squared_error(labels=self.heatmap_batch, predictions=heatmap_voxel_pred_batch)
        self.heatmap_loss = self.heatmap_mid_loss + self.heatmap_voxel_loss
        self.ranking_loss = self._ranking_loss(z_voxel_pred_batch, self.ordinal_batch)
        
        self.loss = self.ranking_loss + 100 * self.heatmap_loss

        with tf.name_scope("loss"):
            tf.summary.scalar("heatmap_mid_loss", self.heatmap_mid_loss)
            tf.summary.scalar("heatmap_voxel_loss", self.heatmap_voxel_loss)
            tf.summary.scalar("heatmap_loss", self.heatmap_loss)
            tf.summary.scalar("ranking_loss", self.ranking_loss)
            tf.summary.scalar("total_loss", self.loss)




