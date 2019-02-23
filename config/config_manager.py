#coding: utf-8

""" 
mpsk's CVDL LayerLibrary @ tf
Version: 2.0

Copyright 2018
"""

import time
import configparser

class ConfigManager(object):
    """ Network Basic Configuration Manager
            with typecheck and structure check
        Support Datatype:   string & float & int & bool & list
                                                        (list_int, list_float, list_plain)

                                (key_list check)
        file => configparser ===================> value_list => model
    """
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.key_list = {}
        self.value_list = {}

    def update(self, section, key, value):
        """ Update a key's value
        """
        if section not in self.value_list.keys():
            self.value_list[section] = {}
        self.value_list[section][key] = value

    def comment(self):
        comm = '\n# mpsk\'s Deep Learning Vision Library\n#\n'
        comm += '# This is a config file created by script\n'
        comm += '# Please note that you should define the data type in model\n'
        comm += '# Parameters in `TRAINING` section will override value in other section\n'
        comm += '# If you want to store a list, please join it with `, `\n#\n'
        comm += '# generated in ' + time.asctime(time.localtime(time.time()))
        comm += '\n\n'
        return comm
    

    def fetch(self, section, key, value):
        return self.value_list[section][key]


    def write(self, filename):
        """ Check and write
            *   If the value keys is in the key_list
            *   If the value can be convert to the desired type
        """
        for kkey in self.key_list.keys():
            if kkey in self.value_list.keys():
                if kkey not in self.config.sections():
                    self.config[kkey] = {}
                for vkey in self.key_list[kkey].keys():
                    if vkey in self.value_list[kkey].keys():
                        if type(self.value_list[kkey][vkey]) == list:
                            self.config[kkey][vkey] = ', '.join(map(lambda x: str(x), self.value_list[kkey][vkey]))
                        else:
                            self.config[kkey][vkey] = self.value_list[kkey][vkey]
                    else:
                        raise KeyError('Failed to save: key `', vkey,'` is not present in value_list!')
            else:
                raise KeyError('Failed to save: section `', kkey,'` is not present in value_list!')
        with open(filename, 'w') as f:
            f.write(self.comment())
            self.config.write(f)
    

    def read(self, filename):
        """ Check and write
            *   If the value keys is in the key_list
            *   If the value can be convert to the desired type
        """
        self.config.read(filename)
        #   if config file contain more keys just ignore
        #   but the keys that the key_list required must be present
        remain_keys = list(set(self.key_list.keys()) - set(self.config.sections()))
        if remain_keys == []:
            for kkey in self.key_list.keys():
                if kkey not in self.value_list.keys():
                    self.value_list[kkey] = {}
                sub_remain_keys = list(set(self.key_list[kkey].keys()) - set(self.config[kkey]))
                if sub_remain_keys == []:
                    for vkey in self.key_list[kkey].keys():
                        type = self.key_list[kkey][vkey]
                        if type == 'int':
                            self.value_list[kkey][vkey] = int(self.config[kkey][vkey])
                        elif type == 'bool':
                            if self.config[kkey][vkey] == 'True':
                                self.value_list[kkey][vkey] = True
                            elif self.config[kkey][vkey] == 'False':
                                self.value_list[kkey][vkey] = False
                            else:
                                raise ValueError('No value option for bool as `', self.config[kkey][vkey], '`')
                        elif type == 'float':
                            self.value_list[kkey][vkey] = float(self.config[kkey][vkey])
                        elif type == 'list_plain':
                            self.value_list[kkey][vkey] = self.config[kkey][vkey].split(', ')
                        elif type == 'list_int':
                            self.value_list[kkey][vkey] = list(map(lambda x: int(x),
                                                                self.config[kkey][vkey].split(', ')))
                        elif type == 'list_float':
                            self.value_list[kkey][vkey] = list(map(lambda x: float(x),
                                                                self.config[kkey][vkey].split(', ')))
                        elif type == 'plain':
                            self.value_list[kkey][vkey] = self.config[kkey][vkey]
                        else:
                            print('Warning: no type called `', type, '`! treated as plain.')
                            self.value_list[kkey][vkey] = self.config[kkey][vkey]
        else:
            raise KeyError('Keys `', remain_keys, '` are unsettled!')

    def bound(self, train_flag=False):
        #   bound to the config
        for kkey in self.value_list.keys():
            if kkey != 'TRAINING':
                for vkey in self.value_list[kkey].keys():
                    setattr(self, vkey, self.value_list[kkey][vkey])
        #   train flag to override the attribute
        if train_flag:
            for vkey in self.value_list['TRAINING'].keys():
                setattr(self, vkey, self.value_list['TRAINING'][vkey])
    

    def print_config(self):
        for kkey in self.value_list.keys():
            print(kkey)
            for vkey in self.value_list[kkey].keys():
                print('\t'+vkey+' : ', self.value_list[kkey][vkey], '\ttype of ', type(self.value_list[kkey][vkey]))
    
    def create_empty(self, filename):
        empty = configparser.ConfigParser()
        for kkey in self.key_list.keys():
            if kkey not in empty.sections():
                empty[kkey] = {}
            for vkey in self.key_list[kkey].keys():
                empty[kkey][vkey] = self.key_list[kkey][vkey]
        with open(filename, 'w') as f:
            f.write(self.comment())
            empty.write(f)
        del empty



if __name__ == '__main__':
    config = ConfigManager()
    config.key_list = {'BASIC':{
                                'alpha': 'int',
                                'beta': 'plain',
                                'gamma': 'float',
                                'theta': 'bool',
                                },
                        'EXTEND':{
                                'alpha': 'int',
                                'beta': 'string',
                                'gamma': 'float',
                                'theta': 'bool',
                                'omega': 'list_int',
                                #'sigma': 'plain'
                                }
                        }
    #'''
    config.update('BASIC', 'alpha', '1')
    config.update('BASIC', 'beta', 'hello')
    config.update('BASIC', 'gamma', '3.1415')
    config.update('BASIC', 'theta', 'True')
    config.update('BASIC', 'omega', 'This won\'t be write into the config')
    config.update('EXTEND', 'alpha', '1')
    config.update('EXTEND', 'beta', 'hello')
    config.update('EXTEND', 'gamma', '3.1415')
    config.update('EXTEND', 'theta', 'True')
    config.update('EXTEND', 'omega', [1,2,3,4,5,6])
    # No support to nested list / tuple
    #config.update('EXTEND', 'sigma', [(2,2), (3,3)])
    config.write('test.cfg')
    config.value_list = {}
    config.read('test.cfg')
    config.print_config()
    #'''
    


