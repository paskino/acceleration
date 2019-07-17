language_level = 3
import cython
import numpy as np
cimport numpy as np

cdef extern int fdiff_for(float *inimage, float *outimage, long nx, long ny, int direction );
cdef extern int fdiff_for_unrolled(float *inimage, float *outimage, long nx, long ny, int direction );
cdef extern int fdiff_for_simd(float *inimage, float *outimage, long nx, long ny, int direction );
cdef extern int fdiff_for_parallel(float *inimage, float *outimage, long nx, long ny, int direction );
cdef extern int fdiff_parallel_whole(float *inimage, float *outimagex, float *outimagey, long nx, long ny);
cdef extern int fdiff_parallel_for_simd(float *inimage, float *outimagex, float *outimagey, long nx, long ny);
cdef extern int fdiff_for_parallel_concurrent(float *inimage, float *outimagex, float *outimagey,  long nx, long ny);

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
    elif version == 3:
        return fdiff_for_parallel(&inimage[0,0], &outimage[0,0], nx, ny, direction)
    
def finite_difference_whole(np.ndarray[np.float32_t, ndim=2, mode="c"] inimage,
                      np.ndarray[np.float32_t, ndim=2, mode="c"] outimagex,
                      np.ndarray[np.float32_t, ndim=2, mode="c"] outimagey,
                      int version):
    cdef int nx, ny
    nx,ny = (<object> inimage).shape
    if version == 0:
        return fdiff_parallel_whole(&inimage[0,0],  &outimagex[0,0], &outimagey[0,0], nx, ny)
    elif version == 1:
        return fdiff_parallel_for_simd(&inimage[0,0],  &outimagex[0,0], &outimagey[0,0], nx, ny)
    elif version == 2:
        return fdiff_for_parallel_concurrent(&inimage[0,0],  &outimagex[0,0], &outimagey[0,0], nx, ny)



def repeat_func(int n, func, *argv):
    cdef int i
    for i in range(n):
        func(*argv)
    return 0