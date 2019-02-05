#coding: utf-8
import sys
import time
import numpy as np


"""
mpsk's CVDL ParameterModel
version 2.0

This class is designed to provide interface to deal the problem of 
nodes and value. Its functions can pack those index-associated values
into understandable key tags. And those functions are with type & key
check.

    Meta Info:
        name*           the name of the model
        author*         the author of this model
        backend*        the backend API that used to train this model
        lib_version*    the library version of LayerLibrary
        in_size*        the input size of the image to this model

        base_lr         base learning rate
        dataset         dataset that the model used
        net_version     the net version of the model
        iters           the number of iterations that the model takes to convergence
        start_time      time string of the start time
        snapshot_time   time string of the snapshot time
        upgraded        the flag of that this model is upgraded from old lib version

        parameters*     the parameter in model
        structure*      TODO:   need to fill with bare script

    (* means the key is essential to the model)

    Index Setup in Layers:
        Bare Convolution 2D
            Convolution weight              :           weights             0

        Batch Normalization Layer
            Batch Normalization Beta        :           beta(offset)        0
            Batch Normalization Gamma       :           gamma(scale)        1
            Batch Normalization Moving Mean :           movmean             2
            Batch Normalization Mov Var     :           movvar              3

        Convolution with bias
            Convolution weight              :           weights             0
            Convolution bias                :           bias                1

        Depthwise Convolution
            Convolution weight              :           weights             0

ChangeLog:
    *   V2.2    Add Meta data (dataset, epoch, upgraded, etc,.)
    *   V2.1a   Bugs Fixed
    *   V2.1    Add meta data to ParameterModelv2
    *   V2.0    Add ParameterModelv2 & TrainableParameterModelv2
"""

class ParameterModelv2(object):
    """ Model Saving Definition 
        Parameter Model version 2
        By mpsk
    """
    def __init__(self, name=None, in_size=368, upgraded=False, epoch=None, dataset=None):
        self.save_dict = {}
        self.save_dict['parameters'] = {}
        self.__paramodel_version__ = '2.1a'
        self.__author__ = 'mpsk'
        self.__backend__ = 'tensorflow'
        self.__net_name__ = name
        self.__in_size__ = in_size
        self.__upgraded__ = upgraded
        self.__epoch__ = epoch
        self.__dataset__ = dataset
        self.index_conf = {  #   Batch Norm Layer
                        'beta':             0,
                        'gamma':            1,
                        'moving_mean':      2,
                        'moving_variance':  3,
                        #   Fully Connected Layer
                        'weight':           0,
                        #   Conv2D Layer
                        'kernel':           0,
                        #   Optional for Conv2D + Bias
                        'bias':             1
                        }
        self.__create_meta__()
    
    def __create_meta__(self):
        #   IMPORTANT KEY
        self.save_dict['name'] = self.__net_name__
        self.save_dict['author'] = self.__author__
        self.save_dict['backend'] = self.__backend__
        self.save_dict['paramodel_version'] = self.__paramodel_version__
        self.save_dict['in_size'] = self.__in_size__
        #   INFORMATIVE KEY
        self.save_dict['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.save_dict['snapshot_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.save_dict['upgraded'] = self.__upgraded__
        self.save_dict['epoch'] = self.__epoch__
        self.save_dict['dataset'] = self.__dataset__
        
    def put_para(self, layer_name, para_name, value):
        """ Common used upgrade model & convert model functions
            put a parameter directly to save_dict
        """
        if layer_name not in self.save_dict['parameters']:
            self.save_dict['parameters'][layer_name] = {}
        self.save_dict['parameters'][layer_name][self.index_conf[para_name]] = value

    def get_para(self, layer_name, para_name, transpose=None):
        """ get the parameter from the save model
        """
        assert self.index_conf[para_name] < len(self.save_dict['parameters'][layer_name])
        if self.save_dict['parameters'] != {}:
            if layer_name in self.save_dict['parameters']:
                if transpose is None:
                    return self.save_dict['parameters'][layer_name][self.index_conf[para_name]]
                else:
                    return np.transpose(
                        self.save_dict['parameters'][layer_name][self.index_conf[para_name]],
                        transpose)
            else:
                raise KeyError('No Parameters found in layer `', layer_name, '`!')
        else:
            raise KeyError('No Parameters found in the `save_dict` in ParameterModel object!')
    
    def load_model(self, path, v1flag=False):
        if v1flag:
            param = np.load(path, encoding='latin1').item()
            self.save_dict['parameters'] = param
        else:
            self.save_dict = np.load(path, encoding='latin1').item()

    def save_model(self, path):
        np.save(path, self.save_dict)


class TrainableParameterModelv2(ParameterModelv2):
    """ Model Saving Definition 
        Trainable Parameter Model version 2
        By mpsk
    """
    def __init__(self,
                name=None,
                in_size=368,
                version='2.2',
                layerlib_version='2.0',
                backend='tensorflow',
                base_lr=None,
                dataset=None,
                upgraded=False):
        super(TrainableParameterModelv2, self).__init__(name)
        self.var_dict = {}
        #   IMPORTANT KEY
        self.save_dict['layerlib_version'] = layerlib_version
        self.save_dict['base_lr'] = base_lr
        self.save_dict['dataset'] = dataset
        self.__in_size__ = in_size
        #   INFORMATIVE KEY
        self.save_dict['net_version'] = version
        self.__create_meta__()

    def check_node(self, layer_name, para_name):
        """ check if the node exist in the model
            if node is not None:
                if there is a value according to the model, then return the existed one
                if not, then return the input one
        """
        return (layer_name, self.index_conf[para_name]) in self.var_dict
    
    def get_node(self, layer_name, para_name):
        return self.var_dict[(layer_name, self.index_conf[para_name])]

    def update_node(self, layer_name, para_name, node):
        """ update reference to node in layer into the var_dict
            if there are duplicated parameters, then re-use it
            NEED BACKEND API SUPPORT
        """
        self.var_dict[(layer_name, self.index_conf[para_name])] = node

    def extract_para(self, sess):
        """ extract paramaters from nodes in model to var_dict
            This is used in a running model
            NEED BACKEND API SUPPORT
        """
        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in self.save_dict['parameters']:
                self.save_dict['parameters'][name] = {}
            self.save_dict['parameters'][name][idx] = var_out

    def save_model(self, sess, path):
        """ save model in a sess
        """
        self.save_dict['parameters'] = {}
        self.extract_para(sess)
        np.save(path, self.save_dict)
        self.save_dict['parameters'] = {}


