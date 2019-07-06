language_level = 3
import cython
import numpy as np
cimport numpy as np

cdef extern int fdiff_for(float *inimage, float *outimage, long nx, long ny, int direction );
cdef extern int fdiff_for_unrolled(float *inimage, float *outimage, long nx, long ny, int direction );
cdef extern int fdiff_for_simd(float *inimage, float *outimage, long nx, long ny, int direction );



def finite_difference(np.ndarray[np.float32_t, ndim=2, mode="c"] inimage,
                      np.ndarray[np.float32_t, ndim=2, mode="c"] outimage,
                      int direction,
                      int version):
#def finite_difference(inimage, outimage , direction):
    cdef int nx, ny
    nx,ny = (<object> inimage).shape 
    if version == 0:
        return fdiff_for(&inimage[0,0], &outimage[0,0], nx, ny, direction)
    elif version == 1:
        return fdiff_for_unrolled(&inimage[0,0], &outimage[0,0], nx, ny, direction)
    elif version == 2:
        return fdiff_for_simd(&inimage[0,0], &outimage[0,0], nx, ny, direction)