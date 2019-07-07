from edo.finite_diff import finite_difference
import numpy
from PIL import Image
import os
import matplotlib.pyplot as plt
import time

data_dir = '.'
which = 'camera.png'

image = numpy.asarray(Image.open(os.path.join(data_dir, which)).convert('L'), dtype=numpy.float32)
print (image.dtype)
dx = image * 0.
dy = image * 0.
dxs = image * 0.
dys = image * 0.
dxu = image * 0.
dyu = image * 0.


t0 = time.time()
finite_difference(image, dx, 0, 0)
finite_difference(image, dy, 1, 0)
t1 = time.time()
finite_difference(image, dxs, 0, 2)
finite_difference(image, dys, 1, 2)
t2 = time.time()
finite_difference(image, dxu, 0, 1)
finite_difference(image, dyu, 1, 1)
t3 = time.time()

print ("Time {} SIMD {} unrolled {}".format(t1-t0,t2-t1,t3-t2))


plt.subplot(2,3,1)
plt.imshow(dx)
plt.subplot(2,3,2)
plt.imshow(dxu)
plt.subplot(2,3,3)
plt.imshow(dxs)
plt.subplot(2,3,4)
plt.imshow(dy)
plt.subplot(2,3,5)
plt.imshow(dyu)
plt.subplot(2,3,6)
plt.imshow(dys)

plt.show()