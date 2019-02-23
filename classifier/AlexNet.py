#coding: utf-8
import os
import tensorflow as tf
import numpy as np
import cv2

from . import Network
from . import Regularizer

"""
Automous Mouse Painting Authentication
    PengFei Wang, Fangrui Liu, Todd Zheng
"""
class AlexNetConfig(Network.NetworkConfig):
    """
        Section
            BASIC
                name
                type
                in_size
                w_summary
                version

                coco_mean_pixel
                batch_size
            TRAINING
                base_lr
                epoch
                epoch_size
                batch_size
            AlexNet
                classes
                drop_out_rate
                mnist_data
    """
    def __init__(self):
        super(AlexNetConfig, self).__init__()
        self.key_list['AlexNet'] = {
            'classes': 'int',
            'drop_out_rate': 'float',
            'mnist_data': 'plain'
        }

class AlexNet(Network.Network):
    def __init__(self, *args, **kwargs):
        """ Init Network parameters
        """
        super(AlexNet, self).__init__(*args, **kwargs)
        self.regularizers = [Regularizer.L2Regularizer(beta=0.001)]

    def build_ph(self):
        self.img = tf.placeholder(tf.float32, shape=[None, self.config.in_size, self.config.in_size])
        self.label = tf.placeholder(tf.float32, shape=[None, self.config.classes])

    def build_train_op(self):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.logit), reduction_indices=[1]))
        with tf.variable_scope("Regularization"):
            self.norm = []
            for regularizer in self.regularizers:
                self.norm.append(regularizer())
        self.loss = cross_entropy + tf.reduce_sum(self.norm) * 0.01
        self.train_step.append(tf.train.GradientDescentOptimizer(self.config.base_lr).minimize(cross_entropy))
        with tf.variable_scope("summaries"):
            self.summ_scalar_list.append(
                tf.summary.scalar("Regression Loss", cross_entropy))
    
    def build_net(self):
        #   
        net= tf.expand_dims(self.img, -1)
        net = self.conv2D_bias_relu(net, 32, 5, 1, 'SAME', name='conv1',
                                    regularizers=self.regularizers, use_loaded=self.load_pretrained, lock=False)
        net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool1')
        net = self.conv2D_bias_relu(net, 32, 5, 1, 'SAME', name='conv2',
                                    regularizers=self.regularizers, use_loaded=self.load_pretrained, lock=False)
        net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool2')
        inputs = tf.layers.dropout(net, rate=self.config.drop_out_rate, training=self.training)
        net = self.conv2D_bias_relu(net, 256, 7, 1, 'VALID', name='fc1',
                                    regularizers=self.regularizers, use_loaded=self.load_pretrained, lock=False)
        flattened = tf.reshape(net, (-1, 256))
        net = self.fullyConnected(flattened, self.config.classes, drop_out=self.config.drop_out_rate, name='fc2',
                                regularizers=self.regularizers, use_loaded=self.load_pretrained, lock=False)
        self.logit = tf.nn.softmax(net)
        if not self.training:
            self.pred_class = tf.argmax(self.logit, axis=1)
            print("- Built pred class tensor!")
    
    def build_monitor(self):
        with tf.variable_scope("accuracy"):
            self.summ_scalar_list.append(tf.summary.scalar("MNIST Accuracy", self.accuracy(self.logit, self.label)))

    def build_summary(self):
        #   merge all summary
        self.summ_scalar = tf.summary.merge(self.summ_scalar_list)
        self.summ_histogram = tf.summary.merge(self.summ_histogram_list)

    def accuracy(self, predictions, labels):
        '''
        Accuracy of a given set of predictions of size (N x n_classes) and
        labels of size (N x n_classes)
        '''
        pred_class = tf.argmax(predictions, axis=1)
        label_class = tf.argmax(labels, axis=1)
        match_mat = tf.equal(pred_class, label_class)
        count_true = tf.count_nonzero(match_mat)
        count_true = tf.cast(count_true, tf.float32)
        return tf.divide(tf.multiply(count_true, 100.0), tf.cast(tf.shape(labels)[0], tf.float32))
    

    def predict(self, img):
        pred = self.sess.run(self.pred_class, feed_dict={self.img: np.array(img).astype(np.float32)/255.0})
        return pred


    def train(self):
        _epoch_count = 0
        _iter_count = 0
        for n in range(self.config.epoch):
            for m in range(self.config.epoch_size):
                batch_xs, batch_ys = mnist.train.next_batch(self.config.batch_size)
                batch_xs = np.reshape(batch_xs, [self.config.batch_size, self.config.in_size, self.config.in_size])
                if n < 10:
                    self.sess.run(self.train_step[0], feed_dict={self.img: batch_xs,
                                                                self.label: batch_ys
                                                            #   add a batch feeder
                                                            })
                print("iter:", _iter_count)
                #   summaries
                if _iter_count % 10 == 0:
                    batch_xs, batch_ys = mnist.train.next_batch(self.config.batch_size)
                    batch_xs = np.reshape(batch_xs, [self.config.batch_size, self.config.in_size, self.config.in_size])
                    #   doing the scalar summary
                    summ_scalar_out,\
                    loss = self.sess.run(
                                [self.summ_scalar, self.loss],
                                                feed_dict={self.img: batch_xs,
                                                            self.label: batch_ys
                                                          #   add a batch feeder
                                                          })
                    for summ in [summ_scalar_out]:
                        self.writer.add_summary(summ, _iter_count)
                    print("epoch ", _epoch_count, " iter ", _iter_count, [loss])
                _iter_count += 1
                self.writer.flush()
            #   doing save numpy params
            self.save_npy()
            _epoch_count += 1
            #   save model every epoch
            if self.log_dir is not None:
                self.saver.save(self.sess, os.path.join(self.log_dir, "model.ckpt"), n)
                pass

        

        