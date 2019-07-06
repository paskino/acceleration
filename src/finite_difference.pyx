language_level = 3
import cython
import numpy as np
cimport numpy as np

cdef extern int fdiff_for(float *inimage, float *outimage, long nx, long ny, int direction );

def finite_difference(np.ndarray[np.float32_t, ndim=2, mode="c"] inimage,
                      np.ndarray[np.float32_t, ndim=2, mode="c"] outimage,
                      int direction):
#def finite_difference(inimage, outimage , direction):
    cdef int nx, ny
    nx,ny = (<object> inimage).shape 
    
    return fdiff_for(&inimage[0,0], &outimage[0,0], nx, ny, direction)