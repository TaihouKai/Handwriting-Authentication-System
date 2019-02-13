# coding: utf-8
import time
import numpy as np
import tensorflow as tf
from .Layers import LayerLibrary
from .Layers import LayerConfig
from . import Regularizer

class NetworkConfig(LayerConfig):
    """
        Section
            BASIC
                name
                type
                in_size
                w_summary
                version
                load_pretrained

                coco_mean_pixel
                batch_size
            TRAINING
                base_lr
                epoch
                epoch_size
                batch_size
    """
    def __init__(self):
        super(NetworkConfig, self).__init__()
        self.key_list['TRAINING']['base_lr'] = 'float'
        self.key_list['TRAINING']['epoch'] = 'int'
        self.key_list['TRAINING']['epoch_size'] = 'int'
        self.key_list['TRAINING']['batch_size'] = 'int'
        self.key_list['BASIC']['coco_mean_pixel'] = 'list_float'
        self.key_list['BASIC']['batch_size'] = 'int'

class Network(LayerLibrary):
    def __init__(self, 
                dataset=None,
                *args,
                **kwargs
                ):
        """
        """
        super(Network, self).__init__(*args, **kwargs)
        #   model log dir control
        if self.training:
            self.log_dir = self.config.name + '_lr' + \
                            str(self.config.base_lr) + \
                            '_insize' + str(self.config.in_size) + '/'
            print(("[*]\tLog dir is : ", self.log_dir))

        self.dataset = dataset

        #   Inside Variable
        self.train_step = []
        self.losses = []

        self.summ_scalar_list = []
        self.summ_accuracy_list = []
        self.summ_image_list = []


    def save_npy(self, save_path=None):
        """ Save the parameters
        WARNING:    Bug may occur due to unknow reason

        :param save_path:       path to save
        :return:
        """
        if save_path == None:
            save_path = self.log_dir + 'model.npy'
        self.para_model.save_model(self.sess, save_path)
        print(("[*]\tfile saved to", save_path))
    
    def restore_sess(self, model=None):
        """ Restore session from ckpt format file

        :param model:   model path
        :return:        Nothing
        """
        if model is not None:
            t = time.time()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, model)
            print("[*]\tSESS Restored!")
        else:
            print("Please input proper model path to restore!")
            raise ValueError

    def BuildModel(self, debug=False):
        """ Building model in tensorflow session

        :return:
        """
        #   input
        tf.reset_default_graph()
        self.sess = tf.Session()
        if self.training:
            self.writer = tf.summary.FileWriter(self.log_dir)
            self.global_step = tf.Variable(0, trainable=False)
            self.build_learningrate()
        with tf.variable_scope('input'):
            self.build_ph()
            print("- PLACEHOLDER build finished!")
        with tf.variable_scope('Regularizers'):
            self.build_regularizers()
        #   assertion
        with tf.variable_scope(self.config.name.split('@')[0]):
            #   the net
            self.build_net()
            print("- NET build finished!")
        if not debug:
            #   initialize all variables
            self.sess.run(tf.global_variables_initializer())
            if self.training:
                #   train op
                self.saver = tf.train.Saver()
                with tf.variable_scope('train'):
                    self.build_train_op()
                    print("- OPTIMIZER & LOSS build finished!")
                with tf.variable_scope('image_summary'):
                    self.build_monitor()
                    self.build_summary()
                    print("- IMAGE_SUMMARY build finished!")
                self.writer.add_graph(self.sess.graph)
        print("[*]\tModel Built")

    def build_ph(self):
        pass

    def build_learningrate(self):
        #   step learning rate policy
        self.learning_rate = tf.train.exponential_decay(self.config.base_lr, self.global_step, 10*self.config.epoch*self.config.epoch_size, 0.333, staircase=True)

    def build_train_op(self):
        pass

    def build_monitor(self):
        pass
    
    def build_net(self):
        raise NotImplementedError
        

    def build_summary(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def build_regularizers(self):
        self.regularizers = [Regularizer.L2Regularizer(beta=1e-4)]
