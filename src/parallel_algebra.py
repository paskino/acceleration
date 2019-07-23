import ctypes
import numpy
import os
import time
from scipy.linalg.blas import daxpy, saxpy


print ("current dir ", os.path.abspath(__file__))
shape = (2048,1024)

A = numpy.float32(2.)
B = numpy.float32(1.)

numpy.random.seed(1)
a = numpy.random.random(shape)
b = numpy.random.random(shape)
#a = 2.5 * numpy.ones(shape)
#b = 1. * numpy.ones(shape)

dll = os.path.abspath(os.path.join( 
         os.path.abspath(os.path.dirname(__file__)),
         'fdiff.dll')
)
print ("dll location", dll)
fdiff = ctypes.cdll.LoadLibrary(dll)


def axpby(a,b,out,A,B,dtype=numpy.float64):


    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_double_p = ctypes.POINTER(ctypes.c_double)

    if a.dtype != dtype:
        a = a.astype(dtype)
    if b.dtype != dtype:
        b = b.astype(dtype)
    
    if dtype == numpy.float32:
        a_p = a.ctypes.data_as(c_float_p)
        b_p = b.ctypes.data_as(c_float_p)
        out_p = out.ctypes.data_as(c_float_p)

    elif dtype == numpy.float64:
        a = a.astype(numpy.float64)
        b = b.astype(numpy.float64)
        a_p = a.ctypes.data_as(c_float_p)
        b_p = b.ctypes.data_as(c_float_p)
        out_p = out.ctypes.data_as(c_float_p)
    else:
        raise TypeError('Unsupported type {}. Expecting numpy.float32 or numpy.float64'.format(dtype))

    out = numpy.empty_like(a)

    
    # int psaxpby(float * x, float * y, float * out, float a, float b, long size)
    fdiff.saxpby.argtypes = [ctypes.POINTER(ctypes.c_float), # pointer to the first array 
                              ctypes.POINTER(ctypes.c_float), # pointer to the second array 
                              ctypes.POINTER(ctypes.c_float), # pointer to the third array 
                              ctypes.c_float,                 # type of A (float)
                              ctypes.c_float,                 # type of B (float)
                              ctypes.c_long]                  # type of size of first array 
    fdiff.daxpby.argtypes = [ctypes.POINTER(ctypes.c_double), # pointer to the first array 
                              ctypes.POINTER(ctypes.c_double), # pointer to the second array 
                              ctypes.POINTER(ctypes.c_double), # pointer to the third array 
                              ctypes.c_double,                 # type of A (c_double)
                              ctypes.c_double,                 # type of B (c_double)
                              ctypes.c_long]                  # type of size of first array 

    if dtype == numpy.float32:
        return fdiff.saxpby(a_p, b_p, out_p, A, B, a.size)
    elif dtype == numpy.float64:
        return fdiff.daxpby(a_p, b_p, out_p, A, B, a.size)


c_float_p = ctypes.POINTER(ctypes.c_float)
c_double_p = ctypes.POINTER(ctypes.c_double)
dtype = numpy.float32
a = a.astype(dtype)
b = b.astype(dtype)
    
if dtype == numpy.float32:
    a_p = a.ctypes.data_as(c_float_p)
    b_p = b.ctypes.data_as(c_float_p)
    out = numpy.empty_like(a)
    out_p = out.ctypes.data_as(c_float_p)
    # int psaxpby(float * x, float * y, float * out, float a, float b, long size)
    fdiff.saxpby.argtypes = [ctypes.POINTER(ctypes.c_float), # pointer to the first array 
                              ctypes.POINTER(ctypes.c_float), # pointer to the second array 
                              ctypes.POINTER(ctypes.c_float), # pointer to the third array 
                              ctypes.c_float,                 # type of A (float)
                              ctypes.c_float,                 # type of B (float)
                              ctypes.c_long]                  # type of size of first array 
    


N = 100
fdiff.saxpby(a_p, b_p, out_p, A, B, a.size)
t0 = time.time()
for i in range(N):
    fdiff.saxpby(a_p, b_p, out_p, A, B, a.size)
t1 = time.time()
print ("saxpby", t1-t0)

axpby(a, b, out, A, B, numpy.float32)
t0 = time.time()
for i in range(N):
    axpby(a, b, out, A, B, numpy.float32)
t1 = time.time()
print ("axpby", t1-t0)

out_numpy = numpy.empty_like(a)
out_numpy2 = numpy.empty_like(a)
numpy.subtract(a,b, out=out_numpy)
t2 = time.time()
for i in range(N):
    numpy.multiply (a, A, out=out_numpy )
    numpy.multiply (b, B, out=out_numpy2)
    numpy.add(out_numpy,out_numpy2, out=out_numpy)
t3 = time.time()
print ("numpy memopt", t3-t2)


t2 = time.time()
for i in range(N):
    out_scipy = saxpy(a,b,a=A)
t3 = time.time()
print ("scipy saxpy", t3-t2)

af = numpy.asfortranarray(a)
bf = numpy.asfortranarray(b)
print ("af.shape", af.shape, "a.shape", a.shape, "af.dtype", af.dtype)
t2 = time.time()
for i in range(N):
    out_scipy_f = daxpy(af,bf,a=A)
t3 = time.time()
print ("scipy saxpy fortran", t3-t2)

out_scipy_f_c = numpy.ascontiguousarray(out_scipy_f)
print ("out_scipy_f.shape", out_scipy_f.shape, "out_scipy_f_c.shape", out_scipy_f_c.shape)
numpy.testing.assert_array_equal(out, out_numpy)
numpy.testing.assert_array_equal(out_numpy, out_scipy)
numpy.testing.assert_array_almost_equal(out_numpy, numpy.ascontiguousarray(out_scipy_f))



