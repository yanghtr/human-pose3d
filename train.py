#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tflearn

from dataset import *
from models import *

# set random seed
seed = 51
tf.set_random_seed(seed)
np.random.seed(seed)

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "./dataset/", " Dataset directory")
flags.DEFINE_string("log_dir", "./log_dir/", "Directory name to save the checkpoints and summaries")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("epochs", 30, "The training epoch")
flags.DEFINE_float("lr_volumeHumanPose", 2.5e-4, "Learning rate of VolumeHumanPose")
FLAGS = flags.FLAGS

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
tf_config.allow_soft_placement=True

def main(_):
    dataLoader = DataLoader(FLAGS.dataset_dir)
    dataset_next_batch = dataLoader.load_batch()
    model = VolumeHumanPose(dataset_next_batch, FLAGS.lr_volumeHumanPose)

    steps_per_epoch = int(dataLoader.img_num / FLAGS.batch_size)

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.epochs):
            for iter in range(steps_per_epoch):
                _, dists, img, joints = sess.run([model.opt_op, model.loss, model.img_batch, model.joints_batch])
                print(dists)
                for i in range(img.shape[0]):
                    visualize_sample(img[i], joints[i]) #, auto_close=True)
                from IPython import embed; embed()



if __name__ == '__main__':
    tf.app.run()

