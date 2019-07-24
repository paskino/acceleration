# acceleration
benchmarking image gradient and numpy

## Benchmarking AXPBY

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
```

As you can see you need to pre allocate 2 arrays of the same size of the output to get to the result. There are 3 loops over all the elements of the arrays being processed. 

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


|method|time (s) 100 iterations|
|--|--|
|fdiff.saxpby| 0.28|
|axpby |0.32|
|intel numpy memopt |0.64|
|intel numpy no memopt |1.36|
|intel scipy saxpy |4.29|
|intel scipy saxpy fortran |2.45|
|numpy| frompyfunc 0.44|
|map |12.87|
|numba| 0.27|
