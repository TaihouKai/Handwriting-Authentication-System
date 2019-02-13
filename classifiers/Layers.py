# coding: utf-8

'''
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms
'''
import sys
import time
import numpy as np
import math
import tensorflow as tf

import sys
sys.path.append('..')
from config.config_manager import ConfigManager

from .ParameterModel import TrainableParameterModelv2, ParameterModelv2
from functools import reduce

""" 
mpsk's CVDL LayerLibrary @ tf
Version: 2.0

    Saving Strategy:
        Saving strategy in this project is under a flatten array form.
        All parameters in this model will be saved according to their names.

    Naming Strategy:
        The basic naming unit of the model is block: take Conv2D+BN+ReLU as example,
        We treat those layers as block, and the default name is settled by its structure
        inside the block. And residual blocks are rigidly named regarding to its
        skipped layer structure so do not try to change it.

    Naming Setup in Block:
        Conv2D_Bias + ReLU:             single layer
        Conv2D + BN + ReLU:             Conv2D(<name>+'_conv') => BN(<name>+'_bn')
        SeperateConv2D + BN + ReLU:     Conv2D_dw(<name>+'_depthwise') => BN(<name>+'_depthwise_bn')
                                        => Conv2D_dw(<name>+'_pointwise') => BN(<name>+'_pointwise_bn') 
        ResidualBlock:                  ---------->res<stage>_branch1------------------------------------------(+)--->
                                            |                                                                   |
                                            -->res<stage>_branch2a-->res<stage>_branch2b-->res<stage>_branch2c---
                                            
        FPN:                            ----->fpn_c<stage>p<stage>-(+)--->fpn_p<stage>------>
                                                                    |
                                        ------->fpn_p<stage+1>-------
Change Log:
    *   V2.7    Configuration Manager   
    *   V2.6    MaskRCNN V1.0
    *   V2.5d   fixed reusing strategy tf.Variable() -> tf.get_variable()
    *   V2.5c   fixed bug caused by data format between tf.constant and tf.get_variable 
                which has made the layer act differently when load the same set of parameter.
    *   V2.5b   remove reshapes in proposal layer and debuging batch norm when loading paras from h5/npy
    *   V2.5a   Py3 upgrade and add new layers(anchors & position associated)
    *   V2.5    Detection Layer tested (ready for demo)
    *   V2.5pre Detection Layer transfered from matterport (need to be tested)   
    *   V2.4    Update proposal layer to match paralleled RPN output
    *   V2.3a   Tested FPN & ResNet50/101 and bug fixed
    *   V2.3    Add FPN setup and fixed bug in ResNet50/101
    *   V2.2    Add ResNet50 and ResNet101  (untested)
    *   V2.1    Add ParameterModelv2 to save a metainfo model
    *   V2.0    Change Naming strategy
    *   V1.4    Add Instance Layer
    *   V1.3    Add Dispatch Layer
    *   V1.2    Add Proposal Layer & RoI Align Layer
    *   V1.1    Add storable BN Layer
    *   V1.0    Numpy Storable Conv Layers (use basic API instead)
    *   V0.2    Put those layers into class
    *   V0.1    TensorFlow Layers
    
NOTE:   ALL LAYERS ACCEPT ONLY (CX, CY, W, H) FORMAT OR (DX, DY, DW, DH) FORMAT
        ANY CONVERTION MUST BE DONE OUTSIDE THE LIBRARY

"""

class LayerConfig(ConfigManager):
    def __init__(self):
        """
        TRAINING section will override some of the argument
        """
        super(LayerConfig, self).__init__()
        self.key_list = {'BASIC':
                                {
                                    'name':'plain',
                                    'type': 'plain',
                                    'in_size':'int',
                                    'w_summary':'bool',
                                    'version':'plain',
                                    'load_pretrained':'bool'
                                },
                        'TRAINING':
                                {
                                }
                        }

#   ======= Net Component ========
class LayerLibrary(object):

    def __init__(self,
                config=None,
                pretrained_model=None,
                training=True):

        config.bound(train_flag=training)
        self.config = config
        self.training = training
        self.check_config_type(config.type)
        self.__version__ = '2.5'
        self.lib_version = self.__version__

        self.pretrained_model = ParameterModelv2('pretrained')
        if pretrained_model is not None:
            self.pretrained_model.load_model(pretrained_model, v1flag=False)
        self.para_model = TrainableParameterModelv2(in_size=self.config.in_size,
                                            name=self.config.name,
                                            version=self.config.version,
                                            layerlib_version=self.__version__)
        self.regularizers = []
        self.summ_histogram_list = []
    
    def check_config_type(self, type):
        """ Override the method to check
        """
        pass

    def _argmax(self, tensor):
        """ ArgMax
        Args:
            tensor	: 2D - Tensor (Height x Width : 64x64 )
        Returns:
            arg		: Tuple of maxlen(self.losses) position
        """
        resh = tf.reshape(tensor, [-1])
        argmax = tf.argmax(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])
    
    #   ============================================
    #   Basic Elements
    #   ============================================

    def batch_norm(self, inputs, decay=0.99, epsilon=1e-3, use_loaded=False, lock=False, is_training=False, reuse=False, name='BatchNorm'):
        """ BatchNormalization Layers
        """
        with tf.variable_scope(name, reuse=reuse):
            para_reuse = 0
            para_ready = False
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if self.para_model.check_node(name, 'beta'):
                # beta = self.para_model.get_node(name, 'beta')
                para_reuse += 1
            if self.para_model.check_node(name, 'gamma'):
                # gamma = self.para_model.get_node(name, 'gamma')
                para_reuse += 1
            if self.para_model.check_node(name, 'moving_mean'):
                # moving_mean = self.para_model.get_node(name, 'moving_mean')
                para_reuse += 1
            if self.para_model.check_node(name, 'moving_variance'):
                # moving_variance = self.para_model.get_node(name, 'moving_variance')
                para_reuse += 1
            if para_reuse == 4:
                print("[*]\treused parameters of ", name)
                para_ready = True
            if use_loaded:
                beta = tf.get_variable('beta', initializer=self.pretrained_model.get_para(name, 'beta'), trainable=(not lock) and is_training)
                gamma = tf.get_variable('gamma', initializer=self.pretrained_model.get_para(name, 'gamma'), trainable=(not lock) and is_training)
                moving_mean = tf.get_variable('moving_mean', initializer=self.pretrained_model.get_para(name, 'moving_mean'), trainable=False)
                moving_variance = tf.get_variable('moving_var', initializer=self.pretrained_model.get_para(name, 'moving_variance'), trainable=False)
                if lock:
                    print("[!]\tLocked ", name + "/BatchNorm", " parameters")
            else:
                beta = tf.get_variable('gamma', initializer=tf.zeros([inputs.get_shape().as_list()[-1]]))
                gamma = tf.get_variable('beta', initializer=tf.ones([inputs.get_shape().as_list()[-1]]))
                moving_mean = tf.get_variable('moving_mean', initializer=tf.zeros([inputs.get_shape().as_list()[-1]]), trainable=False)
                moving_variance = tf.get_variable('moving_var', initializer=tf.ones([inputs.get_shape().as_list()[-1]]), trainable=False)
            if para_ready is not True:
                self.para_model.update_node(name, 'beta', beta)
                self.para_model.update_node(name, 'gamma', gamma)
                self.para_model.update_node(name, 'moving_mean', moving_mean)
                self.para_model.update_node(name, 'moving_variance', moving_variance)

            if is_training:
                batch_mean, batch_variance = tf.nn.moments(inputs, [0, 1, 2], keep_dims=False)

                train_mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay))
                train_variance = tf.assign(moving_variance, moving_variance * decay + batch_variance * (1 - decay))
                with tf.control_dependencies([train_mean, train_variance]):
                    return tf.nn.batch_normalization(inputs, batch_mean, batch_variance, beta, gamma, epsilon)
            else:
                return tf.nn.batch_normalization(inputs, moving_mean, moving_variance, beta, gamma, epsilon)
    
    def fullyConnected(self, inputs, outputs, drop_out=0.0, name='fc', regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Fully Connected Layer
        """
        with tf.variable_scope(name, reuse=reuse):
            para_reuse = 0
            para_ready = False
            inputs = tf.layers.dropout(inputs, rate=drop_out, training=self.training)
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if self.para_model.check_node(name, 'kernel'):
                # kernel = self.para_model.get_node(name, 'kernel')
                para_reuse += 1
            if self.para_model.check_node(name, 'bias'):
                # bias = self.para_model.get_node(name, 'bias')
                para_reuse += 1
            if para_reuse == 2:
                print("[*]\treused parameters of ", name)
                para_ready = True
            if use_loaded:
                kernel = tf.get_variable('kernel', initializer=self.pretrained_model.get_para(name, 'kernel'), trainable=(not lock) and self.training)
                bias = tf.get_variable('bias', initializer=self.pretrained_model.get_para(name, 'bias'), trainable=(not lock) and self.training)
                if lock:
                    print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.get_variable('kernel', initializer=tf.contrib.layers.xavier_initializer(uniform=False)([inputs.get_shape().as_list()[-1], outputs]))
                bias = tf.get_variable('bias', initializer=tf.zeros([outputs]))
            if para_ready is not True:
                self.para_model.update_node(name, 'kernel', kernel)
                self.para_model.update_node(name, 'bias', bias)

            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            net = tf.matmul(inputs, kernel) + bias
            return net
                
    def conv2D(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv', regularizers=[],
               use_loaded=False, lock=False, reuse=False):
        """ Spatial Convolution (CONV2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            conv			: Output Tensor (Convolved Input)
        """
        with tf.variable_scope(name, reuse=reuse):
            if kernel_size is not list:
                kernel_size = [kernel_size, kernel_size]
            else:
                assert len(kernel_size) == 2
            para_reuse = 0
            para_ready = False
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if self.para_model.check_node(name, 'kernel'):
                # kernel = self.para_model.get_node(name, 'kernel')
                para_reuse += 1
            if para_reuse == 1:
                print("[*]\treused parameters of ", name)
                para_ready = True
            if use_loaded:
                kernel = tf.get_variable('kernel', initializer=self.pretrained_model.get_para(name, 'kernel'), trainable=(not lock) and self.training)
                if lock:
                    print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.get_variable('kernel', initializer=tf.contrib.layers.xavier_initializer(uniform=False)(kernel_size+[inputs.get_shape().as_list()[3], filters]))
            if para_ready is not True:
                self.para_model.update_node(name, 'kernel', kernel)
                            
            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            if self.config.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'weights', kernel, collections=['weight']))
            return conv
    
    def conv2D_bias(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv2D_bias', regularizers=[],
                    use_loaded=False, lock=False, reuse=False):
        """ Spatial Convolution (CONV2D) + Bias + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name, reuse=reuse):
            if kernel_size is not list:
                kernel_size = [kernel_size, kernel_size]
            else:
                assert len(kernel_size) == 2
            para_reuse = 0
            para_ready = False
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if self.para_model.check_node(name, 'kernel'):
                # kernel = self.para_model.get_node(name, 'kernel')
                para_reuse += 1
            if self.para_model.check_node(name, 'bias'):
                # bias = self.para_model.get_node(name, 'bias')
                para_reuse += 1
            if para_reuse == 2:
                print("[*]\treused bias parameters of ", name)
                para_ready = True
            if use_loaded:
                kernel = tf.get_variable('kernel', initializer=self.pretrained_model.get_para(name, 'kernel'), trainable=(not lock) and self.training)
                bias = tf.get_variable('bias', initializer=self.pretrained_model.get_para(name, 'bias'), trainable=(not lock) and self.training)
                if lock:
                    print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.get_variable('kernel', initializer=tf.contrib.layers.xavier_initializer(uniform=False)(kernel_size + [inputs.get_shape().as_list()[3], filters]))
                bias = tf.get_variable('bias', initializer=tf.zeros([filters]))
            if para_ready is not True:
                self.para_model.update_node(name, 'kernel', kernel)
                self.para_model.update_node(name, 'bias', bias)
            
            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            conv_bias = tf.nn.bias_add(conv, bias)
            if self.config.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'kernel', kernel, collections=['weight']))
                self.summ_histogram_list.append(tf.summary.histogram(name + 'bias', bias, collections=['bias']))
            return conv_bias
    
    def deconv_shape(self, inputs, kernel_size, stride_size, padding, output_padding=None):
        """ Calculate deconvolution output shape
        """
        assert padding in ['FULL', 'SAME', 'VALID']
        input_shape_list = inputs.shape.as_list()
        output_shape = [input_shape_list[0]]
        for dim_size in input_shape_list[1:3]:
            if inputs is None:
                return None

            # Infer length if output padding is None, else compute the exact length
            if output_padding is None:
                if padding == 'VALID':
                    dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
                elif padding == 'FULL':
                    dim_size = dim_size * stride_size - (stride_size + kernel_size - 2)
                elif padding == 'SAME':
                    dim_size = dim_size * stride_size
            else:
                if padding == 'SAME':
                    pad = kernel_size // 2
                elif padding == 'VALID':
                    pad = 0
                elif padding == 'FULL':
                    pad = kernel_size - 1

                dim_size = ((dim_size - 1) * stride_size + kernel_size - 2 * pad +
                            output_padding)
            output_shape.append(dim_size)
        output_shape = tuple(output_shape+input_shape_list[3:])
        if output_shape[0] is None:
            output_shape = (tf.shape(inputs)[0],) + tuple(output_shape[1:])
            output_shape = tf.stack(list(output_shape))
        return output_shape


    def deconv2D(self, inputs, filters, kernel_size, strides, pad, output_pad=None, name="_deconv", regularizers=[],
                 use_loaded=False, lock=False, reuse=False):
        """ Transposed Convolution 2D (also known as deconvolution 2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            deconv          : Ouput Tensor
        """
        with tf.variable_scope(name, reuse=reuse):
            para_reuse = 0
            para_ready = False
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if self.para_model.check_node(name, 'kernel'):
                # kernel = self.para_model.get_node(name, 'kernel')
                para_reuse += 1
            if para_reuse == 1:
                print("[*]\treused parameters of ", name)
                para_ready = True
            if use_loaded:
                kernel = tf.get_variable('kernel', initializer=self.pretrained_model.get_para(name, 'kernel'), trainable=(not lock) and self.training)
                if lock:
                    print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.get_variable('kernel', initializer=tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, filters, inputs.get_shape().as_list()[3]]))
            if para_ready is not True:
                self.para_model.update_node(name, 'kernel', kernel)
                            
            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)
            
            #   determine the output shape
            if pad == 'VALID':
                output_shape = [1,kernel_size,kernel_size,1] * inputs.shape
            elif pad == 'SAME':
                output_shape = inputs.shape
            else:
                raise ValueError('Invalid Padding type! @', name, ' gate!')
            
            output_shape = self.deconv_shape(inputs, kernel_size, strides, pad, output_pad)

            deconv = tf.nn.conv2d_transpose(inputs, kernel, output_shape, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            if self.config.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'weights', kernel, collections=['weight']))
            return deconv

    def deconv2D_bias(self, inputs, filters, kernel_size, strides, pad, output_pad=None, name="_deconv", regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Dilation 2D (also known as deconvolution 2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            deconv          : Ouput Tensor
        """
        with tf.variable_scope(name, reuse=reuse):
            para_reuse = 0
            para_ready = False
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if self.para_model.check_node(name, 'kernel'):
                # kernel = self.para_model.get_node(name, 'kernel')
                para_reuse += 1
            if self.para_model.check_node(name, 'bias'):
                # bias = self.para_model.get_node(name, 'bias')
                para_reuse += 1
            if para_reuse == 2:
                print("[*]\treused bias parameters of ", name)
                para_ready = True
            if use_loaded:
                kernel = tf.get_variable('kernel', initializer=self.pretrained_model.get_para(name, 'kernel'), trainable=(not lock) and self.training)
                bias = tf.get_variable('bias', initializer=self.pretrained_model.get_para(name, 'bias'), trainable=(not lock) and self.training)
                if lock:
                    print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.get_variable('kernel', initializer=tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size, kernel_size, filters, inputs.get_shape().as_list()[3]]))
                bias = tf.get_variable('bias', initializer=tf.zeros([filters]))
            if para_ready is not True:
                self.para_model.update_node(name, 'kernel', kernel)
                self.para_model.update_node(name, 'bias', bias)
            
            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            # determine the output shape
            output_shape = self.deconv_shape(inputs, kernel_size, strides, pad, output_pad)


            deconv = tf.nn.conv2d_transpose(inputs, kernel, output_shape, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            deconv_bias = tf.nn.bias_add(deconv, bias)
            if self.config.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'kernel', kernel, collections=['weight']))
                self.summ_histogram_list.append(tf.summary.histogram(name + 'bias', bias, collections=['bias']))
            return deconv_bias
    
    def conv2D_depthwise(self, inputs, kernel_size, strides, pad, name="_convdw", regularizers=[],
                         use_loaded=False, lock=False, reuse=False):
        """ Depthwise Spatial Convolution (CONV2D)
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            conv			: Output Tensor
        """
        with tf.variable_scope(name, reuse=reuse):
            if kernel_size is not list:
                kernel_size = [kernel_size, kernel_size]
            else:
                assert len(kernel_size) == 2
            para_reuse = 0
            para_ready = False
            if self.pretrained_model is None:
                use_loaded = False
                print("[!]\tWarning:\tPretrained model not loaded...Using initial value! name: ", name)
            if self.para_model.check_node(name, 'kernel'):
                #kernel = self.para_model.get_node(name, 'kernel')
                para_reuse += 1
            if para_reuse == 1:
                print("[*]\treused parameters of ", name)
                para_ready = True
            if use_loaded:
                kernel = tf.get_variable('kernel', initializer=self.pretrained_model.get_para(name, 'kernel'), trainable=(not lock) and self.training)
                if lock:
                    print("[!]\tLocked ", name, " parameters")
            else:
                kernel = tf.get_variable('kernel', initializer=tf.contrib.layers.xavier_initializer(uniform=False)(kernel_size + [inputs.get_shape().as_list()[3], 1]))
            if para_ready is not True:
                self.para_model.update_node(name, 'kernel', kernel)

            #   Collect Regularization Loss
            if lock is False and regularizers is not None:
                for regularizer in regularizers:
                    regularizer.collect(kernel)

            conv = tf.nn.depthwise_conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            if self.config.w_summary:
                self.summ_histogram_list.append(tf.summary.histogram(name + 'weights', kernel, collections=['weight']))
            return conv

    #   ============================================
    #   Complexes
    #   ============================================

    def conv2D_bn(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv', regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Spatial Convolution (CONV2D) + BatchNormalization
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            conv			: Output Tensor (Convolved Input)
        """
        conv = self.conv2D(inputs, filters, kernel_size, strides, pad, name, regularizers, use_loaded, lock, reuse=reuse)
        norm = self.batch_norm(conv, decay=0.9, epsilon=1e-3, use_loaded=use_loaded, lock=lock, name=name+'_bn', is_training=self.training, reuse=reuse)
        return norm

    def conv2D_bias_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bias_relu', regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Spatial Convolution (CONV2D) + Bias + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            norm			: Output Tensor
        """
        conv_bias = self.conv2D_bias(inputs, filters, kernel_size, strides, pad, name, regularizers, use_loaded, lock, reuse=reuse)
        relu = tf.nn.relu(conv_bias)
        return relu

    def conv2D_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu', regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name):
            #   TODO:   conv2D or conv2D_bias
            conv = self.conv2D(inputs, filters, kernel_size, strides, pad, name, regularizers, use_loaded, lock, reuse=reuse)
            norm = self.batch_norm(conv, decay=0.9, epsilon=1e-3, use_loaded=use_loaded, lock=lock, name=name+'_bn', is_training=self.training, reuse=reuse)
            relu = tf.nn.relu(norm)
            return relu

    def conv2D_bias_bn(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu', regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Spatial Convolution (CONV2D) + Bias + BatchNormalization
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name):
            #   TODO:   conv2D or conv2D_bias
            conv = self.conv2D_bias(inputs, filters, kernel_size, strides, pad, name, regularizers, use_loaded, lock, reuse=reuse)
            norm = self.batch_norm(conv, decay=0.9, epsilon=1e-3, use_loaded=use_loaded, lock=lock, name=name+'_bn', is_training=self.training, reuse=reuse)
            return norm

    def conv2D_bias_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu', regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Spatial Convolution (CONV2D) + Bias + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            norm			: Output Tensor
        """
        with tf.variable_scope(name):
            #   TODO:   conv2D or conv2D_bias
            conv = self.conv2D_bias(inputs, filters, kernel_size, strides, pad, name, regularizers, use_loaded, lock, reuse=reuse)
            norm = self.batch_norm(conv, decay=0.9, epsilon=1e-3, use_loaded=use_loaded, lock=lock, name=name+'_bn', is_training=self.training, reuse=reuse)
            relu = tf.nn.relu(norm)
            return relu
    
    def conv2D_depthwise_bn_relu(self, inputs, kernel_size, strides, pad, name="_convdw_bn_relu", regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Depthwise Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            kernel_size	    : Size of kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        Returns:
            relu			: Output Tensor
        """
        with tf.variable_scope(name):
            conv = self.conv2D_depthwise(inputs, kernel_size, strides, pad, name, regularizers, use_loaded, lock, reuse=False)
            norm = self.batch_norm(conv, decay=0.9, epsilon=1e-3, use_loaded=use_loaded, lock=lock, name=name+'_bn', is_training=self.training, reuse=False)
            relu = tf.nn.relu(norm)
            return relu

    def separable_conv2D(self, inputs, filters, kernel_size, strides, pad, name="_separable_conv", regularizers=[], use_loaded=False, lock=False, reuse=False):
        """ Separable 2D Convolution
        Args:
            inputs			: Input Tensor (Data Type : NHWC)
            filters		    : Number of filters (channels)
            kernel_size     : Size of Kernel
            strides		    : strides
            pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
            name			: Name of the block
            use_loaded      : Use related name to find weight and bias
            lock            : Lock the layer so the parameter won't be optimized
        """
        #   depthwise kernel has the same channel size as the input's channel
        net = self.conv2D_depthwise_bn_relu(inputs, kernel_size, strides, pad, name + '_depthwise', regularizers=regularizers, use_loaded=use_loaded, lock=lock, reuse=reuse)
        #   point-wise should only use 1_1 kernel and 1 strides
        net = self.conv2D_bn_relu(net, filters, 1, 1, pad, name + '_pointwise', regularizers=regularizers, use_loaded=use_loaded, lock=lock, reuse=reuse)
        return net

    #   ============================================
    #   Integrated blocks
    #   ============================================

    def identity_block(self, inputs, filters, kernel_size, strides, stage, block, name='res', regularizers=[], use_loaded=False, lock=False):
        """ Convolutional Block in ResNet Bottleneck setup
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """
        assert len(filters) == 3
        filter_a, filter_b, filter_c = filters
        base_name = name + str(stage) + block + '_branch'
        with tf.variable_scope(base_name):
            net = self.conv2D_bias_bn_relu(inputs, filter_a, 1, 1, 'VALID', base_name+'2a', regularizers, use_loaded, lock)
            net = self.conv2D_bias_bn_relu(net, filter_b, kernel_size, strides, 'SAME', base_name+'2b', regularizers, use_loaded, lock)
            net = self.conv2D_bias_bn(net, filter_c, 1, 1, 'VALID', base_name+'2c', regularizers, use_loaded, lock)
            net = tf.add(net, inputs)
            net = tf.nn.relu(net)
            return net

    def conv_block(self, inputs, filters, kernel_size, strides, stage, block, name='res', regularizers=[], use_loaded=False, lock=False):
        """ Convolutional Block in ResNet Bottleneck setup
        Args:
            inputs	: Input Tensor
            numOut	: Desired output number of channel
            name	: Name of the block
        Returns:
            conv_3	: Output Tensor
        """
        assert len(filters) == 3
        filter_a, filter_b, filter_c = filters
        base_name = name + str(stage) + block + '_branch'
        with tf.variable_scope(base_name):
            net = self.conv2D_bias_bn_relu(inputs, filter_a, 1, strides, 'VALID', base_name+'2a', regularizers, use_loaded, lock)
            net = self.conv2D_bias_bn_relu(net, filter_b, kernel_size, 1, 'SAME', base_name+'2b', regularizers, use_loaded, lock)
            net = self.conv2D_bias_bn(net, filter_c, 1, 1, 'VALID', base_name+'2c', regularizers, use_loaded, lock)
            short_cut = self.conv2D_bias_bn(inputs, filter_c, 1, strides, 'VALID', base_name+'1', regularizers, use_loaded, lock)
            net = tf.add(net, short_cut)
            net = tf.nn.relu(net)
            return net

    #   ============================================
    #   Non-parameter Layers
    #   ============================================
                    
    def crop_and_resize(self, image, boxes, box_ind, crop_size, pad_border=True):
        """
        Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

        Args:
            image: NHWC
            boxes: nx4, x1y1x2y2
            box_ind: (n,)
            crop_size (int):
        Returns:
            n,size,size,C
        """
        assert isinstance(crop_size, int), crop_size
        boxes = tf.stop_gradient(boxes)

        # TF's crop_and_resize produces zeros on border
        if pad_border:
            # this can be quite slow
            image = tf.pad(image, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            boxes = boxes + 1

        def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
            x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

            spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
            spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

            nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
            ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

            nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
            nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)

            return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

        image_shape = tf.shape(image)[1:3]
        boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
        ret = tf.image.crop_and_resize(
            image, boxes, tf.to_int32(box_ind),
            crop_size=[crop_size, crop_size])
        return ret

    def roi_align(self, featuremap, boxes, box_ind, resolution, out_channel, name="RoIAlign"):
        """
        Args:
            featuremap: batchsize x H x W x C
            boxes: batchsize * [Nx4 floatbox]
            resolution: output spatial resolution

        Returns:
            N x res x res x C
        """
        # sample 4 locations per roi bin
        with tf.variable_scope(name):
            ret = self.crop_and_resize(
                        featuremap, 
                        self.cwh2tlbr(boxes),
                        box_ind,
                        resolution * 2)
            ret = tf.nn.avg_pool(ret, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', data_format='NHWC')
            ret = tf.reshape(ret, [-1, resolution, resolution, out_channel])
            return ret

    def log2_graph(self, x):
        """Implementation of Log2. TF doesn't have a native implementation."""
        return tf.log(x) / tf.log(2.0)

    def pyramid_roi_align(self, feature_maps, boxes, image_shape, pool_shape):
        """Implements ROI Pooling on multiple levels of the feature pyramid.

        Params:
        - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]

        Inputs:
        - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                 coordinates. Possibly padded with zeros if not enough
                 boxes to fill the array.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - Feature maps: List of feature maps from different levels of the pyramid.
                        Each is [batch, height, width, channels]

        Output:
        Pooled regions in the shape: [batch, num_boxes, height, width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
        """
        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        pooled = tf.expand_dims(pooled, 0)
        return pooled


    def bbox_transform_inv(self, boxes, deltas, name="bbox_transform_inverse"):
        """ transform bbox with deltas(offsets)
        NOTE:   the bounding box format is in cx, cy, w, h
                the offset format is in dx, dy, dw, dh
        """
        with tf.variable_scope(name):
            cx, cy, w, h = tf.split(boxes, 4, axis=-1)
            dx, dy, dw, dh = tf.split(deltas, 4, axis=-1)
            pred_ctr_x = tf.add(tf.multiply(dx, dw), cx)
            pred_ctr_y = tf.add(tf.multiply(dy, dh), cy)
            pred_w = tf.multiply(tf.exp(dw), w)
            pred_h = tf.multiply(tf.exp(dh), h)
            return tf.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h], axis=-1)

    ############################################################
    #  Batch Associated
    ############################################################

    # ## Batch Slicing
    # Some custom layers support a batch size of 1 only, and require a lot of work
    # to support batches greater than 1. This function slices an input tensor
    # across the batch dimension and feeds batches of size 1. Effectively,
    # an easy way to support batches > 1 quickly with little code modification.
    # In the long run, it's more efficient to modify the code to support large
    # batches and getting rid of this function. Consider this a temporary solution
    def batch_slice(self, inputs, graph_fn, batch_size, names=None):
        """Splits inputs into slices and feeds each slice to a copy of the given
        computation graph and then combines the results. It allows you to run a
        graph on a batch of inputs even if the graph is written to support one
        instance only.

        inputs: list of tensors. All must have the same first dimension length
        graph_fn: A function that returns a TF tensor that's part of a graph.
        batch_size: number of slices to divide the data into.
        names: If provided, assigns names to the resulting tensors.
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        outputs = []
        for i in range(batch_size):
            inputs_slice = [x[i] for x in inputs]
            output_slice = graph_fn(*inputs_slice)
            if not isinstance(output_slice, (tuple, list)):
                output_slice = [output_slice]
            outputs.append(output_slice)
        # Change outputs from a list of slices where each is
        # a list of outputs to a list of outputs and each has
        # a list of slices
        outputs = list(zip(*outputs))

        if names is None:
            names = [None] * len(outputs)

        result = [tf.stack(o, axis=0, name=n)
                  for o, n in zip(outputs, names)]
        if len(result) == 1:
            result = result[0]
        return result

    def batch_flatten(self, input, batch_size):
        """
        Gives the tag that describes the batch index of a sample.
        and a flatten tensor which is shaped to
            (batch_size*rois, ?)
        :param input:       (batch_size, rois, ?)
        :param batch_size:  int
        :return tags:       (batch_size*rois)
        :return out:        (batch_size*rois, ?)
        """
        tags = []
        out = []
        for n in range(batch_size):
            tag = n * tf.ones((tf.shape(input)[1]), tf.int32)
            tags.append(tag)
            out.append(input[n])
        tags = tf.concat(tags, 0)
        out = tf.concat(out, 0)
        return tags, out

    def batch_rebuild(self, input, tags, batch_size):
        """
        reshape tensor by batch according the tag of each sample
            (batch_size, rois, w, h, c)
        :param input:       (batch_size*rois, ?)
        :param batch_size:  int
        :return tags:       (batch_size*rois)
        :return out:        (batch_size, rois, ?)
        """
        out = []
        for n in range(batch_size):
            indices = tf.where(tf.equal(tags, n))
            indices = tf.squeeze(indices)
            out.append(tf.gather(input, indices))
        out = tf.stack(out, 0)
        return out

    ############################################################
    #  Proposal Layer
    ############################################################

    def apply_box_deltas_graph(self, boxes, deltas):
        """Applies the given deltas to the given boxes.
        boxes: [N, (y1, x1, y2, x2)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        """
        # Convert to y, x, h, w
        height = boxes[:, 2] - boxes[:, 0]
        width = boxes[:, 3] - boxes[:, 1]
        center_y = boxes[:, 0] + 0.5 * height
        center_x = boxes[:, 1] + 0.5 * width
        # Apply deltas
        center_y += deltas[:, 0] * height
        center_x += deltas[:, 1] * width
        height *= tf.exp(deltas[:, 2])
        width *= tf.exp(deltas[:, 3])
        # Convert back to y1, x1, y2, x2
        y1 = center_y - 0.5 * height
        x1 = center_x - 0.5 * width
        y2 = y1 + height
        x2 = x1 + width
        result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
        return result

    def clip_boxes_graph(self, boxes, window):
        """
        boxes: [N, (y1, x1, y2, x2)]
        window: [4] in the form y1, x1, y2, x2
        """
        # Split
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
        clipped.set_shape((clipped.shape[0], 4))
        return clipped

    def proposal_layer_mrcnn(self, scores, deltas, anchors, proposal_count, nms_threshold, bbox_std_dev, name="Proposal_mrcnn"):
        """Receives anchor scores and selects a subset to pass as proposals
        to the second stage. Filtering is done based on anchor scores and
        non-max suppression to remove overlaps. It also applies bounding
        box refinement deltas to anchors.

        Inputs:
            rpn_probs: [batch, anchors, (bg prob, fg prob)]
            rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
            anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates

        Returns:
            Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
        """
        with tf.variable_scope(name):
            # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
            scores = scores[:, :, 1]
            # Box deltas [batch, num_rois, 4]
            deltas = deltas * np.reshape(bbox_std_dev, [1, 1, 4])

            # Improve performance by trimming to top anchors by score
            # and doing the rest on the smaller subset.
            pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[1])
            ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                             name="top_anchors").indices
            scores = self.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                       self.config.batch_size)
            deltas = self.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                       self.config.batch_size)
            pre_nms_anchors = self.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                                self.config.batch_size,
                                                names=["pre_nms_anchors"])

            # Apply deltas to anchors to get refined anchors.
            # [batch, N, (y1, x1, y2, x2)]
            boxes = self.batch_slice([pre_nms_anchors, deltas],
                                      lambda x, y: self.apply_box_deltas_graph(x, y),
                                      self.config.batch_size,
                                      names=["refined_anchors"])

            # Clip to image boundaries. Since we're in normalized coordinates,
            # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
            window = np.array([0, 0, 1, 1], dtype=np.float32)
            boxes = self.batch_slice(boxes,
                                      lambda x: self.clip_boxes_graph(x, window),
                                      self.config.batch_size,
                                      names=["refined_anchors_clipped"])

            # Filter out small boxes
            # According to Xinlei Chen's paper, this reduces detection accuracy
            # for small objects, so we're skipping it.

            # Non-max suppression
            def nms(boxes, scores):
                indices = tf.image.non_max_suppression(
                    boxes, scores, proposal_count,
                    nms_threshold, name="rpn_non_max_suppression")
                proposals = tf.gather(boxes, indices)
                # Pad if needed
                padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
                proposals = tf.pad(proposals, [(0, padding), (0, 0)])
                return proposals

            proposals = self.batch_slice([boxes, scores], nms,
                                          self.config.batch_size)
            return proposals

    def proposal_layer(self, score_map, bbox_offset, pos_map_single, anchors_per_pixel, nms_iou_threshold=0.5, pre_nms_resrv=200, post_nms_resrv=16, nms_score_threshold=float('-inf'), bbox_std_dev=None, name="ProposalLayer"):
        """ extract bbox from score map
        NOTE:   return format shape in [batch_num, num_true, (cx, cy, w, h) 
        Args:
            score_map       :   list of tensor which has shape of (batch_size, width, height, 2*anchors)
                                [negative, positive]
            bbox_offset     :   list of tensor which has shape of (batch_size, width, height, 4*anchors)
                                [dx, dy, log(dw), log(dh)]
            pos_map_single  :   list of tensor which has shape of (1, width, height, 2) (re-used)
            anchors         :   list of anchors
            nms_iou_threshold   :   Non Maximum Suppression score threshold
            pre_nms_resrv   :   Number of record that the layer reserved before NMS
            post_nms_resrv  :   Number of record that the layer reserved after NMS

        Return:
            layer_out       :   list of tensor in different shapes

        NOTE:
            To implement with dynamic batch_size we use a rebundant channel to store all node
            We declare a pre-define style network structure, and control with the tf.shape
            with input and output

            Be aware of tf.shape, it can generate tensor of unknown shapes, and generate dence
            tensor in graph. Just take care of it after pad using tf.shape
        """
        with tf.variable_scope(name):
            #assert len(score_map) == len(pos_map_single) == len(bbox_offset)
            concat_ex_bbox_offset_pad = []
            concat_ex_score_pad = []
            with tf.variable_scope("Tensor_Reshape"):
                '''
                #   list of tensor 
                for n in range(len(score_map)):
                    out_size = score_map[n].shape[1:3]
                    #   (batch_size, out_size, out_size, anchors, 4)
                    #   multiply pos_map to batch_size
                    pos_map = tf.tile(pos_map_single[n], multiples=[tf.shape(score_map[n])[0], 1, 1, 1, 1])
                    #   transform_inv(pos_map, offset) = abs_position
                    ex_bbox_offset = self.bbox_transform_inv(pos_map, self.tlbr_rev(bbox_offset[n]))
                    #   (batch_size, out_size * out_size * anchors， 4)
                    ex_bbox_offset = tf.reshape(ex_bbox_offset,
                                                (-1, out_size[0] * out_size[1] * anchors_per_pixel, 4))
                    if bbox_std_dev is not None:
                        assert len(bbox_std_dev) == 4
                        ex_bbox_offset = ex_bbox_offset * np.reshape(bbox_std_dev, [1, 1, 4])
                    #   positive only <class num 1>
                    #   (batch_size, out_size * out_size, anchors)
                    score= tf.reshape(score_map[n], [-1, out_size[0]*out_size[1]*anchors_per_pixel, 2])
                    score = score[:, :, 1]
                    #   (batch_size, out_size * out_size * anchors)
                    #   expand to defined batch_size
                    ex_score_pad = tf.pad(score, [[0, self.config.batch_size-tf.shape(score_map[n])[0]],[0,0]])
                    ex_score_pad = tf.reshape(ex_score_pad, (self.config.batch_size, out_size[0]*out_size[1]*anchors_per_pixel))
                    ex_bbox_offset_pad = tf.pad(ex_bbox_offset, [[0, self.config.batch_size-tf.shape(score_map[n])[0]],[0,0],[0,0]])
                    #   concat_ex_bbox_offset_pad is the concatenated tensor
                    #   concat_ex_score_pad is the concatenated tensor
                    concat_ex_score_pad.append(ex_score_pad)
                    concat_ex_bbox_offset_pad.append(ex_bbox_offset_pad)
                    print(out_size)
                #   concatenate all tensors
                concat_ex_score_pad = tf.concat(concat_ex_score_pad, 1)
                concat_ex_bbox_offset_pad = tf.concat(concat_ex_bbox_offset_pad, 1)
                '''
                pos_map = tf.tile(pos_map_single, multiples=[tf.shape(score_map)[0], 1, 1])
                #   transform_inv(pos_map, offset) = abs_position
                ex_bbox_offset = self.bbox_transform_inv(pos_map, self.tlbr_rev(bbox_offset))
                concat_ex_bbox_offset_pad = ex_bbox_offset
                concat_ex_score_pad = score_map[:,:,1]
            #   TODO:   shape check before the `RoI Extract` stage
            assert concat_ex_score_pad.shape[1] == concat_ex_bbox_offset_pad.shape[1] \
                                                == reduce(lambda x, y: x+y,
                                                            [x*x*anchors_per_pixel for x in [int(math.ceil(
                                                                                    self.config.in_size/float(2**x))) for x in range(2, 7)]]
                                                          )
            #   Create nodes for every batch in mini-batch
            #   And you need to collect those Nodes
            #   layer_out have length of batch_size
            layer_out = []
            with tf.variable_scope("RoI_Extract"):
                for idx in range(self.config.batch_size):
                    with tf.variable_scope("TopK_NMS_Cell"):
                        batch_ind = idx * tf.ones((concat_ex_score_pad.shape[1]))
                        #   Find Top-k element index
                        _, topk_indx = tf.nn.top_k(concat_ex_score_pad[idx], k=pre_nms_resrv, sorted=True)
                        #   Collect all value (bbox, score, batch_indx)
                        topk_bbox = tf.gather(concat_ex_bbox_offset_pad[idx], topk_indx)
                        topk_score = tf.gather(concat_ex_score_pad[idx], topk_indx)
                        topk_batch_ind = tf.gather(batch_ind, topk_indx)
                        topk_bbox = tf.squeeze(topk_bbox)
                        #   Get the index after doing NMS
                        topk_idx_nms = tf.image.non_max_suppression(self.cwh2tlbr_rev(topk_bbox),
                                                                    topk_score,
                                                                    post_nms_resrv,
                                                                    iou_threshold=nms_iou_threshold,
                                                                    score_threshold=nms_score_threshold)
                        #   Collect result after NMS
                        #   (num_true, 4)
                        nms_bbox = tf.gather(topk_bbox, topk_idx_nms)
                        nms_score = tf.gather(topk_score, topk_idx_nms)
                        #   (num_true, 1)
                        nms_batch_ind = tf.gather(topk_batch_ind, topk_idx_nms)
                        #   (num_true, 5) [cx, cy, w, h, batch_ind]
                        t = tf.concat([nms_bbox,
                                        tf.expand_dims(nms_score, axis=-1),
                                        tf.expand_dims(nms_batch_ind, axis=-1)], -1)
                        layer_out.append(t)
                layer_out = tf.stack(layer_out, 0)
                #   return at input batch_size
                #   (batch_size, num_true, 5)
                return layer_out, nms_score

    ############################################################
    #  Detection Layer
    ############################################################

    def refine_detections_graph(self, rois, probs, deltas, window, bbox_std_dev, min_conf=0.7,
                                max_instance=100, nms_threshold=0.3):
        """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                    bounding box deltas.
            window: (y1, x1, y2, x2) in image coordinates. The part of the image
                that contains the image excluding the padding.

        Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)] where
            coordinates are normalized.
        """
        # Class IDs per ROI
        class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
        # Class probability of the top class of each ROI
        indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(deltas, indices)
        # Apply bounding box deltas
        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = self.apply_box_deltas_graph(
            rois, deltas_specific * bbox_std_dev)
        # Clip boxes to image window
        refined_rois = self.clip_boxes_graph(refined_rois, window)

        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if min_conf:
            conf_keep = tf.where(class_scores >= min_conf)[:, 0]
            keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                            tf.expand_dims(conf_keep, 0))
            keep = tf.sparse_tensor_to_dense(keep)[0]

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois, keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            """Apply Non-Maximum Suppression on ROIs of the given class."""
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=max_instance,
                iou_threshold=nms_threshold)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            # Pad with -1 so returned tensors have the same shape
            gap = max_instance - tf.shape(class_keep)[0]
            class_keep = tf.pad(class_keep, [(0, gap)],
                                mode='CONSTANT', constant_values=-1)
            # Set shape so map_fn() can infer result shape
            class_keep.set_shape([max_instance])
            return class_keep

        # 2. Map over class IDs
        nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                             dtype=tf.int64)
        # 3. Merge results into one list, and remove -1 padding
        nms_keep = tf.reshape(nms_keep, [-1])
        nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
        # 4. Compute intersection between keep and nms_keep
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(nms_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]
        # Keep top detections
        roi_count = max_instance
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are normalized.
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        gap = max_instance - tf.shape(detections)[0]
        detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
        return detections

    def detection_layer(self, rois, mrcnn_class, mrcnn_bbox, window, image_shape,
                        bbox_std_dev, max_instance=100, min_conf=0.7, nms_threshold=0.3,
                        name="DetectionLayer"):
        """Takes classified proposal boxes and their bounding box deltas and
        returns the final detection boxes.

        Returns:
        [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
        coordinates are normalized.
        """
        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        window = self.norm_boxes_graph(window, image_shape)

        # Run detection refinement graph on each item in the batch
        detections_batch = self.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: self.refine_detections_graph(x, y, w, z,
                                                            bbox_std_dev, min_conf,
                                                            max_instance, nms_threshold),
            self.config.batch_size)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.batch_size, max_instance, 6])

    ############################################################
    #  Target Layer
    ############################################################

    def bb_intersection_over_union(self, bboxes1, bboxes2):
        """ Numpy ndarray iou implementation
        """
        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = tf.maximum(x11, tf.transpose(x21))
        yA = tf.maximum(y11, tf.transpose(y21))
        xB = tf.minimum(x12, tf.transpose(x22))
        yB = tf.minimum(y12, tf.transpose(y22))

        # compute the area of intersection rectangle
        interArea = tf.maximum((xB - xA + 1), 0) * tf.maximum((yB - yA + 1), 0)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)

        return iou

    def overlaps_graph(self, boxes1, boxes2):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        """
        # 1. Tile boxes2 and repeat boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.
        # TF doesn't have an equivalent to np.repeat() so simulate it
        # using tf.tile() and tf.reshape.
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute intersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps


    def cwh2tlbr(self, bbox, name="CWH2TLBR_GATE"):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        with tf.variable_scope(name):
            cx, cy, w, h = tf.split(bbox, 4, axis=-1)
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            return tf.concat([x1, y1, x2, y2], axis=-1, name=name+"_out")

    def tlbr_rev(self, bbox, name="TLBRREV_GATE"):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        with tf.variable_scope(name):
            x1, y1, x2, y2 = tf.split(bbox, 4, axis=-1)
            return tf.concat([y1, x1, y2, x2], axis=-1, name=name+"_out")

    def cwh2tlbr_rev(self, bbox, name="CWH2TLBR_GATE"):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        with tf.variable_scope(name):
            cx, cy, w, h = tf.split(bbox, 4, axis=-1)
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            return tf.concat([y1, x1, y2, x2], axis=-1, name=name+"_out")

    def tlbr2cwh(self, bbox, name="TLBR2CWH_GATE"):
        """ Cx, Cy, W, H to TopLeft BottomRight
        """
        with tf.variable_scope(name):
            x1, y1, x2, y2 = tf.split(bbox, 4, axis=-1)
            w = tf.abs(x2 - x1)
            h = tf.abs(y2 - y1)
            cx = x1 + w/2
            cy = y1 + h/2
            return tf.concat([cx, cy, w, h], axis=-1, name=name+"_out")


    def norm_boxes_graph(self, boxes, shape):
        """Converts boxes from pixel coordinates to normalized coordinates.
        boxes: [..., (cx, cy, w, h)] in pixel coordinates
        shape: [..., (height, width)] in pixels
        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.
        Returns:
            [..., (cx, cy, w, h)] in normalized coordinates
        """
        h, w = tf.split(tf.cast(shape, tf.float32), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
        shift = tf.constant([0., 0., 1., 1.])
        return tf.divide(boxes - shift, scale)


    def denorm_boxes_graph(self, boxes, shape):
        """Converts boxes from normalized coordinates to pixel coordinates.
        boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
        shape: [..., (height, width)] in pixels
        Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
        coordinates it's inside the box.
        Returns:
            [..., (y1, x1, y2, x2)] in pixel coordinates
        """
        h, w = tf.split(tf.cast(shape, tf.float32), 2)
        scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
        shift = tf.constant([0., 0., 1., 1.])
        return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)

    def patch_with_crop_and_resize(self, img, s_bbox, out_size, target_size, name="ReProjGate"):
        """ back projection according to bounding box
        Args:
            img     :   img has shape of (h, w, c)
            s_bbox  :   bounding box shape of (4,) [cx, cy, w, h]
        
        TODO:   This Layer should output lossless target size output but now we just uses
                upsampled output. This need to be fixed in the future!!
        """
        with tf.variable_scope(name):
            #   first resize the image to its origin size
            with tf.variable_scope(name+"_image_resize"):
                s_bbox = tf.cast(s_bbox, tf.int32)
                s_mask = tf.cond(
                                tf.logical_and(tf.greater(s_bbox[2],0), tf.greater(s_bbox[3], 0)),
                                lambda: tf.image.resize_images(img, tf.stack([s_bbox[3], s_bbox[2]])),
                                lambda: tf.zeros([out_size, out_size, tf.shape(img)[-1]])
                                )

            #   convert to top left - bottom right format
            s_bbox_tlbr = self.cwh2tlbrtlbr(s_bbox, name=name+"_CWH2TLBR")

            #   then crop the valid area on origin image
            with tf.variable_scope(name+"_image_crop"):
                crop_box = tf.concat([tf.nn.relu(-s_bbox_tlbr[:2]), tf.nn.relu(s_bbox[2:4]-tf.nn.relu(s_bbox_tlbr[2:4]-(out_size, out_size)))], 0)
                #crop_box = tf.cast(crop_box, tf.int32)
                crop_box = tf.concat([tf.minimum(crop_box[:2], crop_box[2:4]), tf.maximum(crop_box[:2], crop_box[2:4])], 0)
                s_mask = s_mask[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
                s_mask = tf.cond(tf.logical_and(tf.greater(tf.shape(s_mask)[0],0), tf.greater(tf.shape(s_mask)[1],0)),
                                    lambda: s_mask,
                                    lambda: tf.zeros([out_size, out_size, tf.shape(img)[-1]], dtype=tf.float32)
                                    )
            #   finally pad the output into original size
            #   s_mask has shape of (out_size, out_size)
            with tf.variable_scope(name+"_image_pad"):
                padd_left = tf.nn.relu(s_bbox_tlbr[0])
                padd_right = tf.nn.relu(out_size-s_bbox_tlbr[2])
                padd_top = tf.nn.relu(s_bbox_tlbr[1])
                padd_bottom = tf.nn.relu(out_size-s_bbox_tlbr[3])
                padd = tf.stack([[padd_top, padd_bottom], [padd_left, padd_right], [0, 0]], name="padd_param")
                s_mask = tf.pad(s_mask, padd)
            s_mask = tf.image.resize_images(s_mask,(target_size, target_size))
            return s_mask

    def dispatch_layer(self, bboxes, masks, batch_ind, batch_size, rois_max, out_size, target_size, name="DispatchLayer"):
        """ Dispatch Layer
            dispatch every instance into different channel
        Args:
            bboxes      :   bboxes from PPN (batch * rois, 4) in format of (cx, cy, w, h)
            masks       :   masks from Mask Net (batch * rois, size, size, channel)
            batch_ind   :   batch_ind from PPN (batch * rois)
            batch_size  :   images in a mini-batch

        """
        with tf.variable_scope(name):
            #   re-map the segmentation to origin image
            #   match bbox & segmentation, then crop
            #   1. first we need to pad the input to batch_size * rois
            with tf.variable_scope('ReProjection_Layer'):
                bboxes_pad = tf.pad(bboxes, [[0, tf.constant(batch_size*rois_max)-tf.shape(bboxes)[0]],[0,0]])
                bboxes_pad = tf.cast(bboxes_pad, tf.int32)
                bboxes_pad = tf.reshape(bboxes_pad, [batch_size*rois_max, 4])
                masks_pad = tf.pad(masks, [[0, tf.constant(batch_size*rois_max)-tf.shape(bboxes)[0]],[0,0],[0,0],[0,0]])
                masks_pad = tf.reshape(masks_pad, [batch_size*rois_max, out_size, out_size, 2])
                fan_in = []
                for n in range(batch_size * rois_max):
                    #   s_mask has shape of (height, width, channel)
                    s_mask = self.patch_with_crop_and_resize(
                                    masks_pad[n], 
                                    bboxes_pad[n],
                                    out_size,
                                    target_size,
                                    name="ReProjGate_"+str(n))
                    fan_in = tf.expand_dims(s_mask, axis=0)
                fan_in = tf.concat(fan_in, 0)
                #   (rois, out_size, out_size, channel)
                #   fan_in has shape rois * height * width * channel
                fan_in = fan_in[:tf.shape(bboxes)[0]]

            #   2. secondly, we match masks to each batch
            with tf.variable_scope("Dispatch_Layer"):
                batch_mask = []
                for n in range(batch_size):
                    ind = tf.where(tf.equal(batch_ind, n))
                    ind = tf.squeeze(ind)
                    #   t has shape of (instance channel, height, width, channel)
                    #   if there is no RoI in this batch, fill with zeros
                    t = tf.cond(tf.greater(tf.squeeze(tf.shape(ind)), 0),
                                lambda: tf.gather(fan_in, ind),
                                lambda: tf.zeros(tf.shape(fan_in)))
                    #   if there are spare channels, fill them
                    t = tf.cond(tf.less(tf.shape(t)[0], rois_max), 
                                lambda: tf.pad(t, [[0,rois_max-tf.shape(t)[0]],[0,0],[0,0],[0,0]]),
                                lambda: t[:rois_max])
                    batch_mask.append(tf.expand_dims(t, axis=0))
                batch_mask = tf.concat(batch_mask, 0)
            #   (batch_size, out_size, out_size, instance channel, channel)
            batch_mask = tf.transpose(batch_mask, perm=[0, 2, 3, 4, 1])
            batch_mask = tf.reshape(batch_mask, [batch_size, out_size, out_size, rois_max, 2])
            return batch_mask, fan_in

    def makeGaussian(self, height, width, bbox):
        """ Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = tf.range(0, width, 1, dtype=tf.float32)
        y = tf.expand_dims(tf.range(0, height, 1, dtype=tf.float32), axis=1)
        x0 = bbox[1]
        y0 = bbox[0]
        sigma = tf.maximum(bbox[2], bbox[3])/2
        return tf.exp(-4.0 * tf.log(2.0) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def instance_layer(self, centers, batch_ind, batch_size, rois_max, out_size, name="DispatchLayer"):
        """ Instance Layer 
        Put a guassian responce to each channel according to heatmap response
        Args:
            centers :   (cx, cy)
        """
        with tf.variable_scope(name):
            #   1. first we need to pad the input to batch_size * rois  
            with tf.variable_scope('ReProjection_Layer'):
                centers_pad = tf.pad(centers, [[0, tf.constant(batch_size*rois_max)-tf.shape(centers)[0]],[0,0]])
                centers_pad = tf.reshape(centers_pad, [batch_size*rois_max, 4])
                fan_in = []
                for n in range(batch_size * rois_max):
                    #   s_mask has shape of (height, width, 2)
                    s_mask_pos = tf.expand_dims(self.makeGaussian(out_size, out_size, centers_pad[n]), axis=-1)
                    '''
                    s_mask_neg = 1 - s_mask_pos
                    #   s_mask = (1-s_mask, s_mask)
                    s_mask = tf.concat([s_mask_neg, s_mask_pos], -1)
                    '''
                    fan_in.append(tf.expand_dims(s_mask_pos, axis=0))
                fan_in = tf.concat(fan_in, 0)
                #   (rois, out_size, out_size, channel)
                #   fan_in has shape rois * height * width * channel
                fan_in = fan_in[:tf.shape(centers)[0]]

            #   2. secondly, we match map to each batch
            with tf.variable_scope("Dispatch_Layer"):
                batch_mask = []
                for n in range(batch_size):
                    ind = tf.where(tf.equal(batch_ind, n))
                    ind = tf.squeeze(ind, axis=-1)
                    debug = ind
                    #   t has shape of (instance channel, height, width, channel)
                    #   if there is no RoI in this batch, fill with zeros
                    t = tf.cond(tf.greater(tf.squeeze(tf.shape(ind)), 0),
                                lambda: tf.gather(fan_in, ind),
                                lambda: tf.zeros(tf.shape(fan_in)))
                    #   if there are spare channels, fill them
                    t = tf.cond(tf.less(tf.shape(t)[0], rois_max), 
                                lambda: tf.pad(t, [[0,rois_max-tf.shape(t)[0]],[0,0],[0,0],[0,0]]),
                                lambda: t[:rois_max])
                    batch_mask.append(tf.expand_dims(t, axis=0))
                batch_mask = tf.concat(batch_mask, 0)
            #   (batch_size, out_size, out_size, instance channel, 2)
            batch_mask = tf.transpose(batch_mask, perm=[0, 2, 3, 4, 1])
            batch_mask = tf.reshape(batch_mask, [batch_size, out_size, out_size, rois_max, 1])
            return batch_mask, debug

    # ========================
    #   Big Network Components
    # ========================
                
    def feature_pyramid_network(self, inputs, topdown_pyr_size=256, super_downsample=True, use_loaded=False, lock=False):
        """ Feature Pyramid Network
        Args:
            inputs              : Input tensor
            feature_extrator    : Feature extrator the net uses
            topdown_pyr_size    : Out channel size of FPN
            finetune            : flag to train feature extractor
            use_loaded          : flag whether to load FPN parameters
            lock                : flag whether lock the FPN parameter
        Return:
            net                 : list of Output tensor
        """
        assert len(inputs) in [4, 5]
        with tf.variable_scope("FPN"):
            pyramid = []
            last_stage = None
            bottom = True
            #   Connection block
            for stage in range(len(inputs))[::-1]:
                if stage == 0:
                    continue
                stage_str = str(stage+1)
                if bottom is True:
                    net = self.conv2D_bias(inputs[stage], topdown_pyr_size, 1, 1, 'SAME', 
                                            name='fpn_c'+stage_str+'p'+stage_str, use_loaded=use_loaded, lock=lock)
                    bottom = False
                else:
                    assert last_stage is not None
                    net = self.conv2D_bias(inputs[stage], topdown_pyr_size, 1, 1, 'SAME', 
                                            name='fpn_c'+stage_str+'p'+stage_str, use_loaded=use_loaded, lock=lock)
                    #   upsampling to the last stage to current size
                    last_stage = tf.image.resize_images(last_stage, [inputs[stage].shape[1], inputs[stage].shape[2]], 
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    net = tf.add(last_stage, net)
                last_stage = net
                #   
                net = self.conv2D_bias(net, topdown_pyr_size, 3, 1, 'SAME', 
                                            name='fpn_p'+stage_str, use_loaded=use_loaded, lock=lock)
                pyramid.append(net)
            if super_downsample:
                pyramid = [tf.contrib.layers.max_pool2d(pyramid[0], [2,2], [2,2], padding='SAME', 
                                                    scope='fpn_p'+str(len(inputs)+1))] + pyramid
            #   returned list contained (layers + 1) transformed features
            #   P2, P3, P4, P5[, P6]
            print("[*]\tFPN Built!")
            return pyramid[::-1]

    def feature_extractor(self, inputs, net_type='VGG', stage5=False, finetune=False, conv4_2_split=False):
        """ Feature Extractor
        For VGG Feature Extractor down-scale by x8
        For ResNet Feature Extractor downscale by x8 (Current Setup)

        Net use VGG as default setup
        Args:
            inputs      : Input Tensor (Data Format: NHWC)
            name        : Name of the Extractor
        Returns:
            net         : Output Tensor            
        """
        if conv4_2_split is True and net_type is not 'VGG':
            raise ValueError('`conv4_2_split` option only available in VGG!')
        if 'ResNet' in net_type:
            #   different setup in ResNet50/101
            assert net_type in ["ResNet50", "ResNet101"]
            block_setup = {"ResNet50": 5,
                           "ResNet101": 22}
            with tf.variable_scope(net_type):
                net = tf.pad(inputs, [[0,0],[3,3],[3,3],[0,0]])
                feat_l1 = net = self.conv2D_bias_bn_relu(net, 64, 7, 2, 'VALID', 'conv1', use_loaded=True, lock=not finetune)
                net = tf.contrib.layers.max_pool2d(net, [3,3], [2,2], padding='SAME', scope='pool1')
                #   down scale by 2 
                #   stage 2
                net = self.conv_block(net, [64, 64, 256], 3, 1, stage=2, block='a', use_loaded=True, lock=not finetune)
                net = self.identity_block(net, [64, 64, 256], 3, 1, stage=2, block='b', use_loaded=True, lock=not finetune)
                net = self.identity_block(net, [64, 64, 256], 3, 1, stage=2, block='c', use_loaded=True, lock=not finetune)
                feat_l2 = net       #   /(2^2)
                #   down scale by 2
                #   stage 3
                net = self.conv_block(net, [128, 128, 512], 3, 2, stage=3, block='a', use_loaded=True, lock=not finetune)
                net = self.identity_block(net, [128, 128, 512], 3, 1, stage=3, block='b', use_loaded=True, lock=not finetune)
                net = self.identity_block(net, [128, 128, 512], 3, 1, stage=3, block='c', use_loaded=True, lock=not finetune)
                net = self.identity_block(net, [128, 128, 512], 3, 1, stage=3, block='d', use_loaded=True, lock=not finetune)
                feat_l3 = net       #   /(2^3)
                #   down scale by 2 /8
                #   stage 4
                net = self.conv_block(net, [256, 256, 1024], 3, 2, stage=4, block='a', use_loaded=True, lock=not finetune)
                for i in range(block_setup[net_type]):
                    net = self.identity_block(net, [256, 256, 1024], 3, 1, stage=4, block=chr(98+i), use_loaded=True,
                                              lock=not finetune)
                feat_l4 = net       #   /(2^4)
                #   down scale by 2 
                #   stage 5
                if stage5:
                    net = self.conv_block(net, [512, 512, 2048], 3, 2, stage=5, block='a', use_loaded=True, lock=not finetune)
                    net = self.identity_block(net, [512, 512, 2048], 3, 1, stage=5, block='b', use_loaded=True,
                                              lock=not finetune)
                    net = self.identity_block(net, [512, 512, 2048], 3, 1, stage=5, block='c', use_loaded=True,
                                              lock=not finetune)
                    feat_l5 = net   #   /(2^5)
                    print("[*]\tLoaded ", net_type, "of 5 stage!")
                    return [feat_l1, feat_l2, feat_l3, feat_l4, feat_l5]
                else:
                    print("[*]\tLoaded ", net_type, "of 4 stage!")
                    return [feat_l1, feat_l2, feat_l3, feat_l4]
        elif net_type == 'VGG':
            with tf.variable_scope(net_type):
                #   VGG based
                net = self.conv2D_bias_relu(inputs, 64, 3, 1, 'SAME', 'conv1_1', use_loaded=True, lock=not finetune)
                net = feat_l1 = self.conv2D_bias_relu(net, 64, 3, 1, 'SAME', 'conv1_2', use_loaded=True, lock=not finetune)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool1')
                #   down scale by 2
                net = self.conv2D_bias_relu(net, 128, 3, 1, 'SAME', 'conv2_1', use_loaded=True, lock=not finetune)
                net = feat_l2 = self.conv2D_bias_relu(net, 128, 3, 1, 'SAME', 'conv2_2', use_loaded=True, lock=not finetune)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool2')
                #   down scale by 2
                net = self.conv2D_bias_relu(net, 256, 3, 1, 'SAME', 'conv3_1', use_loaded=True, lock=not finetune)
                net = self.conv2D_bias_relu(net, 256, 3, 1, 'SAME', 'conv3_2', use_loaded=True, lock=not finetune)
                net = self.conv2D_bias_relu(net, 256, 3, 1, 'SAME', 'conv3_3', use_loaded=True, lock=not finetune)
                net = feat_l3 = self.conv2D_bias_relu(net, 256, 3, 1, 'SAME', 'conv3_4', use_loaded=True, lock=not finetune)
                net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool3')
                #   down scale by 2
                net = self.conv2D_bias_relu(net, 512, 3, 1, 'SAME', 'conv4_1', use_loaded=True, lock=not finetune)
                net = self.conv2D_bias_relu(net, 512, 3, 1, 'SAME', 'conv4_2', use_loaded=True, lock=not finetune)
                if conv4_2_split:
                    return [feat_l1, feat_l2, feat_l3, net]
                net = self.conv2D_bias_relu(net, 512, 3, 1, 'SAME', 'conv4_3', use_loaded=True, lock=not finetune)
                net = feat_l4 = self.conv2D_bias_relu(net, 512, 3, 1, 'SAME', 'conv4_4', use_loaded=True, lock=not finetune)
                if stage5:
                    net = self.conv2D_bias_relu(net, 512, 3, 1, 'SAME', 'conv5_1', use_loaded=True, lock=not finetune)
                    net = self.conv2D_bias_relu(net, 512, 3, 1, 'SAME', 'conv5_2', use_loaded=True, lock=not finetune)
                    net = self.conv2D_bias_relu(net, 512, 3, 1, 'SAME', 'conv5_3', use_loaded=True, lock=not finetune)
                    feat_l5 = self.conv2D_bias_relu(net, 512, 3, 1, 'SAME', 'conv5_4', use_loaded=True, lock=not finetune)
                    print("[*]\tLoaded ", net_type, "of 5 stage!")
                    return [feat_l1, feat_l2, feat_l3, feat_l4, feat_l5]
                else:
                    print("[*]\tLoaded ", net_type, "of 4 stage!")
                    return [feat_l1, feat_l2, feat_l3, feat_l4]
        elif net_type == 'MobileNet_V1':
            with tf.variable_scope(net_type):
                feat_l1 = self.conv2D_bn_relu(inputs, 32, 3, 2, 'SAME', 'Conv2d_0', use_loaded=True, lock=not finetune)
                net = self.separable_conv2D(feat_l1, 64, 3, 1, 'SAME', 'Conv2d_1', use_loaded=True, lock=not finetune)
                feat_l2 = self.separable_conv2D(net, 128, 3, 2, 'SAME', 'Conv2d_2', use_loaded=True, lock=not finetune)
                net = self.separable_conv2D(feat_l2, 128, 3, 1, 'SAME', 'Conv2d_3', use_loaded=True, lock=not finetune)
                feat_l3 = self.separable_conv2D(net, 256, 3, 2, 'SAME', 'Conv2d_4', use_loaded=True, lock=not finetune)
                net = self.separable_conv2D(feat_l3, 256, 3, 1, 'SAME', 'Conv2d_5', use_loaded=True, lock=not finetune)
                net = self.separable_conv2D(net, 512, 3, 1, 'SAME', 'Conv2d_6', use_loaded=True, lock=not finetune)
                net = self.separable_conv2D(net, 512, 3, 1, 'SAME', 'Conv2d_7', use_loaded=True, lock=not finetune)
                feat_l4 = self.separable_conv2D(net, 512, 3, 1, 'SAME', 'Conv2d_8', use_loaded=True, lock=not finetune)
                if stage5:
                    print("[!]\tWarining: Stage5 flag is not avaliable in MobileNet backbone!")
                print("[*]\tLoaded ", net_type, "of 4 stage!")
                return [feat_l1, feat_l2, feat_l3, feat_l4]
        raise TypeError("No matched model structure to type \'"+ net_type+"\'!")

