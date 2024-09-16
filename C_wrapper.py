import ctypes
import numpy as np
import time

perlin_lib_mp = ctypes.CDLL('./naive_perlin_mp.so')
perlin_lib = ctypes.CDLL('./naive_perlin.so')

perlin_lib_mp.perlin_noise.argtypes = [
    ctypes.POINTER(ctypes.c_int),    
    ctypes.c_int,                    
    ctypes.c_int,                    
    ctypes.c_float,                  
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float)  
]

perlin_lib.perlin_noise.argtypes = [
    ctypes.POINTER(ctypes.c_int),    
    ctypes.c_int,                    
    ctypes.c_int,                    
    ctypes.c_float,                  
    ctypes.POINTER(ctypes.c_float), 
    ctypes.POINTER(ctypes.c_float)  
]


perlin_lib_mp.perlin_noise.restype = None
perlin_lib.perlin_noise.restype = None

def init_C(perm, grads, w,h):
    output = np.zeros((w, h), dtype=np.float32)
    perm = np.array(perm, dtype=np.int32)
    grads = np.array(grads, dtype=np.float32)
    return perm, grads, output

def C_perlin_noise_mp(perm, w,h, scale, grads):
    output = np.zeros((w, h), dtype=np.float32)
    
    perm = np.array(perm, dtype=np.int32)
    grads = np.array(grads, dtype=np.float32)

    perlin_lib_mp.perlin_noise(
        perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),  
        ctypes.c_int(w),                                   
        ctypes.c_int(h),                                   
        ctypes.c_float(scale),                            
        grads.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))  
    )
    return output

def C_perlin_noise(perm, w,h, scale, grads):
    output = np.zeros((w, h), dtype=np.float32)
    
    perm = np.array(perm, dtype=np.int32)
    grads = np.array(grads, dtype=np.float32)

    perlin_lib.perlin_noise(
        perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),  
        ctypes.c_int(w),                                   
        ctypes.c_int(h),                                   
        ctypes.c_float(scale),                            
        grads.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),  
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))  
    )
    return output