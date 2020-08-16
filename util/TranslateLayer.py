# coding: utf-8
"""
HWAT Project
Copyright 2019
"""
import json
import numpy as np
from functools import reduce

class TranslateLayer():
    """
    Translate the data content and help the algorithm communicating with other module
    The translate layer accept output whose type is numpy array or (nested) list of ndarray
    - Method:
        * encode
        * decode
    """
    def __init__(self):
        pass
    
    def _recursive_toist(self, data):
        """
        Internal recursive to_list function
        it convert any shape of nested list of ndarray to list
        and easy to serialize
        """
        if type(data) == list or type(data) == tuple:
            data = list(map(lambda x: self._recursive_toist(x), list(data)))
        elif type(data) == np.ndarray:
            data = data.tolist()
        else:
            print("[!]\tIgnored type `" + str(type(data)) + "` which does not belong to `list`, `tuple` or `numpy.ndarray`")
            return None
        return data

    def __shape__(self, data):
        if type(data) == int or type(data) == float:
            return 1
        if type(data) == list or type(data) == tuple:
            return len(data)
        elif type(data) == np.ndarray:
            return data.shape
        elif data == None:
            return None
        else:
            raise TypeError("Type `" + str(type(data)) + "` is not supported!")
    
    def _recursive_ndarray_stop_cond(self, li):
        not_equal = False
        for elem in li:
            if self.__shape__(elem) != self.__shape__(li[0]):
                not_equal = True
        return not not_equal

    def _recursive_ndarray(self, data):
        """
        Internal recurisive function which builds up a ndim array
        it helps ndarray to rebuild the deserialized tensor
        """
        if type(data) == float or type(data) == int or type(data) == np.ndarray:
            return True, data
        elif type(data) == list or type(data) == tuple:
            ret, data = list(map(list, zip(*map(lambda x: self._recursive_ndarray(x), list(data)))))
            ret = reduce(lambda a, b: a and b, ret)
            '''
            print("<<< ", data, "  ", ret)
            #print(reduce(lambda a, b: a and b, map(lambda x: self.__shape__(x)==self.__shape__(data[0]), data)))
            print(self._recursive_ndarray_stop_cond(data))
            print("shape: ", list(map(lambda x: self.__shape__(data[0]), list(data))))
            #'''
            if reduce(lambda a, b: a and b, map(lambda x: self.__shape__(x)==self.__shape__(data[0]), data)) and ret:
                data = np.array(data)
                return True, data
            else:
                return False, data
        else:
            print("[!]\tIgnored type `" + str(type(data)) + "` which does not belong to `list`, `tuple` or `numpy.ndarray`")
            return False, None

    
    def serialize(self, data):
        """ External API for translate layer
        """
        return json.dumps(self._recursive_toist(data))

    def deserialize(self, data):
        return self._recursive_ndarray(json.loads(data))[1]




if __name__ == '__main__':
    a = [[[np.array([1, 2, 3]), np.array([4, 5, 6])], np.array([4, 5])]]
    trsltlyr = TranslateLayer()
    da = trsltlyr.serialize(a)
    lda = trsltlyr.deserialize(da)
    print(lda)