#pragma once
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "dll_export.h"

DLL_EXPORT int fdiff_for(float *inimage, float *outimage, long nx, long ny, int direction);
DLL_EXPORT int fdiff_for_unrolled(float *inimage, float *outimage, long nx, long ny, int direction);
DLL_EXPORT int fdiff_for_simd(float *inimage, float *outimage, long nx, long ny, int direction);
DLL_EXPORT int fdiff_for_parallel(float *inimage, float *outimage, long nx, long ny, int direction);
DLL_EXPORT int fdiff_parallel_for_simd(float *inimage, float *outimagex, float *outimagey, long nx, long ny);
DLL_EXPORT int fdiff_parallel_whole(float *inimage, float *outimagex, float *outimagey, long nx, long ny);
DLL_EXPORT int fdiff_for_parallel_concurrent(float *inimage, float *outimagex, float *outimagey, long nx, long ny);