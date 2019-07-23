import ctypes
import numpy
import os
import time

print ("current dir ", os.path.abspath(__file__))
shape = (2048,2048, 128)

A = numpy.float32(2.)
B = numpy.float32(1/5.)

numpy.random.seed(1)
a = numpy.random.random(shape)
b = numpy.random.random(shape)

dll = os.path.abspath(os.path.join( 
         os.path.abspath(os.path.dirname(__file__)),
         'fdiff.dll')
)
print ("dll location", dll)
fdiff = ctypes.cdll.LoadLibrary(dll)

c_float_p = ctypes.POINTER(ctypes.c_float)
a = a.astype(numpy.float32)
b = b.astype(numpy.float32)
out = numpy.empty_like(a)

a_p = a.ctypes.data_as(c_float_p)
b_p = b.ctypes.data_as(c_float_p)
out_p = out.ctypes.data_as(c_float_p)

# int psaxpby(float * x, float * y, float * out, float a, float b, long size)
fdiff.psaxpby.argtypes = [ctypes.POINTER(ctypes.c_float), # pointer to the first array 
                          ctypes.POINTER(ctypes.c_float), # pointer to the second array 
                          ctypes.POINTER(ctypes.c_float), # pointer to the third array 
                          ctypes.c_float,                 # type of A (float)
                          ctypes.c_float,                 # type of B (float)
                          ctypes.c_long]                  # type of size of first array 


N = 10
fdiff.psaxpby(a_p, b_p, out_p, A, B, a.size)
t0 = time.time()
for i in range(N):
    fdiff.psaxpby(a_p, b_p, out_p, A, B, a.size)
t1 = time.time()
print ("parallel_add", t1-t0)

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

numpy.testing.assert_array_equal(out, out_numpy)



