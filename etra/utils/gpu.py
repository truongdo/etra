#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Truong Do <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

try:
    from pynvml import *
except:
    print "Warning: pyvnml is not available, cannot detect GPU card automatically"

def get_gpu_available():
    gpu = -1 
    try:
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        max_mem_free = -1
        
        for i in range(0, deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            memInfo = nvmlDeviceGetMemoryInfo(handle)
            mem_total = memInfo.total 
            mem_used = memInfo.used 
            mem_free = memInfo.free
            
            if mem_free > max_mem_free:
                max_mem_free = mem_free
                gpu = i
    finally:
        return gpu

if __name__ == "__main__":
    print get_gpu_available()
