#include "parallel_algebra_acc.h"

DLL_EXPORT int saxpby(float *restrict x, float *restrict y, float *restrict out, float a, float b, long size){
    long i = 0;

#pragma acc data copyin(x[0:size], y[0:size]) copy( out[0:size])
{
//#pragma acc kernels present(x,y,out)
#pragma acc parallel loop present(x,y,out)
    for (i=0; i < size; i++)
    {
        *(out + i ) = a * ( *(x + i) ) + b * ( *(y + i) ); 
    }
}

#pragma acc exit data delete(x[0:size],y[0:size])
    return 0;
    
}

DLL_EXPORT int daxpby(double * x, double * y, double * out, double a, double b, long size) {
	long i = 0;
#pragma acc data copyin(x[0:size], y[0:size]) copy( out[0:size])

	{
#pragma acc kernels present(x,y,out)
		for (i = 0; i < size; i++)
		{
			*(out + i) = a * (*(x + i)) + b * (*(y + i));
		}
	}
#pragma acc exit data delete(x[0:size],y[0:size])
	return 0;

}


