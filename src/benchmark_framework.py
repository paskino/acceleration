from edo.finite_diff import finite_difference, finite_difference_whole, repeat_func
import numpy
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
from ccpi.framework import TestData
from ccpi.optimisation.operators import FiniteDiff


data_dir = '.'
which = 'camera.png'
size = (2048,2048)
#size = (512,512)
loader = TestData()
data = loader.load(TestData.CAMERA , size=size)
fdx = FiniteDiff(data.geometry)
fdy = FiniteDiff(data.geometry, direction=1)
outdatay = data * 0.
outdatax = data * 0.


image = numpy.asarray(Image.open(os.path.join(data_dir, which)).convert('L').resize(size),
        dtype=numpy.float32)
image = data.as_array()
print (image.dtype, image.shape)
dx = image * 0.
dy = image * 0.
dxs = image * 0.
dys = image * 0.
dxu = image * 0.
dyu = image * 0.
dxp = image * 0.
dyp = image * 0.
dxw = image * 0.
dyw = image * 0.
dxps = image * 0.
dyps = image * 0.

N = 1000
t0 = time.time()
for i in range(N):
    finite_difference(image, dx, 0, 0)
    finite_difference(image, dy, 1, 0)
# t1a = time.time()
# repeat_func(N,finite_difference,image, dx,0,0)
# repeat_func(N,finite_difference,image, dy,1,0)
t1 = time.time()
# print ("Baseline Python", t1a-t0, 'Cython', t1-t1a)
print ("Baseline", t1-t0)
for i in range(N):
    finite_difference(image, dxs, 0, 2)
    finite_difference(image, dys, 1, 2)
t2 = time.time()
print ("#pragma omp simd", t2-t1)
for i in range(N):
    finite_difference(image, dxu, 0, 1)
    finite_difference(image, dyu, 1, 1)
    
t3 = time.time()
print ("unrolled", t3-t2)
for i in range(N):
    finite_difference(image, dxp, 0, 3)
    finite_difference(image, dyp, 1, 3)
t4 = time.time()
print ("#pragma omp parallel", t4-t3)
for i in range(N):
    #fd.direction = 0
    fdx.direct(data, out=outdatax)
    #fd.direction = 1
    fdy.direct(data, out=outdatay)
t5 = time.time()
print ("numpy", t5-t4)
for i in range(N):
    finite_difference_whole(image, dxw, dyw, 0)
t6 = time.time()
print ("finite_difference_whole method 0", t6-t5)
for i in range(N):
    finite_difference_whole(image, dxps, dyps, 2)
t7 = time.time()
print ("parallel concurrent", t7-t6)

#print ("Baseline {}\nSIMD {}\nunrolled {}\nParallel {}\nFramework {}\nparallel whole {}".format(
#    t1-t0,t2-t1,t3-t2,t4-t3,t5-t4, t6-t5))


plt.subplot(2,5,1)
plt.imshow(dx)
plt.subplot(2,5,2)
plt.imshow(dxu)
plt.subplot(2,5,3)
plt.imshow(dxs)
plt.subplot(2,5,4)
plt.imshow(dxp)
plt.subplot(2,5,5)
plt.imshow(outdatay.as_array())
plt.subplot(2,5,6)
plt.imshow(dy)
plt.subplot(2,5,7)
plt.imshow(dyu)
plt.subplot(2,5,8)
plt.imshow(dys)
plt.subplot(2,5,9)
plt.imshow(dyp)
plt.subplot(2,5,10)
plt.imshow(outdatax.as_array())

plt.show()

plt.subplot(1,2,1)
plt.imshow(dxw)
plt.subplot(1,2,2)
plt.imshow(dyw)
plt.show()