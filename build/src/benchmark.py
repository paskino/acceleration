from edo.finite_diff import finite_difference
import numpy
from PIL import Image
import os
import matplotlib.pyplot as plt
import time

data_dir = '.'
which = 'camera.png'

image = numpy.asarray(Image.open(os.path.join(data_dir, which)).convert('L'), dtype=numpy.float32)
print (image.dtype, image.shape)
dx = image * 0.
dy = image * 0.
dxs = image * 0.
dys = image * 0.
dxu = image * 0.
dyu = image * 0.
dxp = image * 0.
dyp = image * 0.

N = 1000
t0 = time.time()
for i in range(N):
    finite_difference(image, dx, 0, 0)
    finite_difference(image, dy, 1, 0)
t1 = time.time()
for i in range(N):
    finite_difference(image, dxs, 0, 2)
    finite_difference(image, dys, 1, 2)
t2 = time.time()
for i in range(N):
    finite_difference(image, dxu, 0, 1)
    finite_difference(image, dyu, 1, 1)
t3 = time.time()
for i in range(N):
    finite_difference(image, dxp, 0, 3)
    finite_difference(image, dyp, 1, 3)
t4 = time.time()

print ("Time {}\nSIMD {}\nunrolled {}\nParallel {}".format(t1-t0,t2-t1,t3-t2,t4-t3))


plt.subplot(2,4,1)
plt.imshow(dx)
plt.subplot(2,4,2)
plt.imshow(dxu)
plt.subplot(2,4,3)
plt.imshow(dxs)
plt.subplot(2,4,4)
plt.imshow(dxp)
plt.subplot(2,4,5)
plt.imshow(dy)
plt.subplot(2,4,6)
plt.imshow(dyu)
plt.subplot(2,4,7)
plt.imshow(dys)
plt.subplot(2,4,8)
plt.imshow(dyp)

plt.show()