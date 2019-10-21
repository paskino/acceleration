# acceleration
Benchmarking different tools to speed up computationally intensive code in Python.

## Benchmarking AXPBY

Code to be found [here](https://github.com/paskino/acceleration/blob/master/src/parallel_algebra.py)
Core numpy array operations are well optimised for single thread operations and there is no benefit of using multithreading (like OpenMP) to do the basic algebric operations `+-*/power minimum maximum`. Comparisons to be added.

Doing multiple operations like `a*X + b*Y` where `a,b` are scalars and `X,Y` are numpy arrays can in fact benefit from multithreading. 

### Pure numpy

```python
# numpy no memopt, aka no memory pre-allocation
out = a*X + b*Y

# numpy memopt
out = numpy.empty_like(X)
tmp = numpy.empty_like(X)

numpy.multiply(X,a,out=out)
numpy.multiply(Y,b,tmp)
out += tmp

# frompyfunc
n_axpby = numpy.frompyfunc(lambda x,y,a,b: a*x + b*y, 4,1)
out = n_axpby(X,Y,a,b)
```

As you can see you need to pre allocate 2 arrays of the same size of the output to get to the result. There are 3 loops over all the elements of the arrays being processed. 

### Python map

One could also use the `map` function from Python. 

```python

from itertools import chain

out_map =  numpy.fromiter(
    chain.from_iterable(
        map(lambda var: a * var[0] + b * var[1], zip(X,Y))), dtype=a.dtype
	).reshape(a.shape)

```

Notice that here the `lambda` is getting the constants `a,b` from the parent scope. So it's not quite reccommended to use this method.

### C/OpenMP

Creating a simple C/OpenMP function to do the `axpby` is relatively easy. It can be used with `ctypes` (or `cython`).

```c
DLL_EXPORT int saxpby(float * x, float * y, float * out, float a, float b, long size){
    long i = 0;
#pragma omp parallel
{
#pragma omp for
    for (i=0; i < size; i++)
    {
        *(out + i ) = a * ( *(x + i) ) + b * ( *(y + i) ); 
    }
}
    return 0;
    
}

DLL_EXPORT int daxpby(double * x, double * y, double * out, double a, double b, long size) {
	long i = 0;
#pragma omp parallel
	{
#pragma omp for
		for (i = 0; i < size; i++)
		{
			*(out + i) = a * (*(x + i)) + b * (*(y + i));
		}
	}
	return 0;

}
```

There are 2 advantages here: 
1. there is need only to allocate the memory for the result, 
2. there is only one cycle over the elements of the arrays
However, the little C code must be compiled as shared library.

Using this simple C/OpenMP function can be done with `ctypes`. Here we propose 2 version, 
1. [`axpby`](https://github.com/paskino/acceleration/blob/master/src/parallel_algebra.py#L31) where the call to the function is done from within a Python function where the right types are chosen
2. a simple version where the user selects the right `ctypes argtypes` before the call. 

The difference between the 2 methods is that in 1 all calls will select the proper function to call (float or double). In 2 the appropriate types for the C functions have been established before the call. This part is pretty [long](https://github.com/paskino/acceleration/blob/master/src/parallel_algebra.py#L34L71) and not particularly pythonesque. Clearly in method 1 there is an overhead because for each call we execute som 40 lines of (Python) code before the actual real call. 

### BLAS (SciPy)

Using a BLAS from SciPy may be beneficial. However, the main disadvantage here is that the arrays must be Fortran-contiguous which will reduce the speed, if you work with C-contiguous arrays. Additionally BLAS implement `axpy` : `a*X+Y` rather than `axpby`.

```python
from scipy.linalg.blas import saxpy, daxpy
# X,Y c contiguous arrays
out = saxpy(X,Y,a=a)

# notice that you need to use daxpy in this case
af = numpy.asfortranarray(X)
bf = numpy.asfortranarray(Y)
out = daxpy(af,bf,a=a)

# the output C-contiguous
out_C = numpy.ascontiguousarray(out)
```
### Numba

Lastly a very simple implementation with `jit` and `prange` (parallel range) from [`numba`](https://numba.pydata.org/)

```python
from numba import jit, prange

@jit(nopython=True)
def numba_axpby(x,y,out,ca,cb):
    for i in prange(x.size):
        out.flat[i] = ca*x.flat[i] + cb*y.flat[i]

out = numpy.empty_like(X)
numba_axpby(X,Y,out,a,b)


```

### Results

I've run tests on 2 machines with different characteristics. Do not compare across columns.
Also the results are very dependent on the size of the problem. 

Especially the numba implementation on Linux seems to have large overhead for small problems.

|method|Windows time (s) 100 iterations|Linux time (s) 100 iterations 2048**2 |
|--|--|--|
|fdiff.saxpby| 0.28| 0.15 |
|axpby |0.32| 0.18 |
|intel numpy memopt |0.64| 1.04 |
|intel numpy no memopt |1.36| 1.31 |
|intel scipy saxpy |4.29| 4.15 |
|intel scipy saxpy fortran |2.45| 3.76 |
|numpy frompyfunc | 0.44| 1.23 |
|map |12.87| 64.33 |
|numba| 0.27| 0.69 |
