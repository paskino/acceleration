#include "parallel_algebra_acc.h"
#include <stdlib.h>     /* malloc, free, rand */
#include <stdio.h>
#include <time.h>
#include <omp.h>


int set_vector(float * v, long size, float value){

    for (long i=0; i<size; i++){
        * ( v + i )  = value;
    }

    return 0;
}

int test_result (float * v, long size, float value){
    long res = 0;
    for (long i=0; i<size; i++){
        res += * ( v + i ) - value < 0.0001 ? 0 : 1;
    }
    printf("First and last values: %f, %f\n", *v, *(v+size-1));
    if (res == 0){
        return 0;
    } else {
        return res;
    }
    return 0;
    
}

int serial_saxpby(float * x, float * y, float * out, float a, float b, long size){
    for (long i=0; i < size; i++)
        {
            *(out + i ) = a * ( *(x + i) ) + b * ( *(y + i) ); 
        }
    return 0;
}

int omp_saxpby(float * x, float * y, float * out, float a, float b, long size){
    //omp_set_num_threads(15);
#pragma omp parallel
    {
#pragma omp for
        for (long i=0; i < size; i++)
            {
                *(out + i ) = a * ( *(x + i) ) + b * ( *(y + i) ); 
            }
        
    }
    return 0;
}

double elapsed_time(struct timespec * start, struct timespec * finish){

    double elapsed;

    elapsed = (finish->tv_sec - start->tv_sec);
    elapsed += (finish->tv_nsec - start->tv_nsec) / 1000000000.0;

    return elapsed;
}

int main(int argc, char ** argv){

    int Mb = 100;
    if (argc >= 1)
        sscanf(argv[1], "%d", &Mb);
    printf("Test on %d Mb\n", Mb);


    int size = 1024 * 1024 * Mb;
    int N=10;
    clock_t before, after;
    int msec;
    struct timespec start, finish;
    double dt;
    long res[3];

    float *restrict v1 = (float *) malloc(size * sizeof(float));
    if ( v1 == NULL ){
        exit (-1);
    }

    float *restrict v2 = (float *) malloc(size * sizeof(float));
    if ( v2 == NULL ){
        exit (-1);
    }

    set_vector(v1, size, 1);
    set_vector(v2, size, -2);

    float a = 1.1;
    float b = 1.41;

    float *restrict out = (float *) malloc(size * sizeof(float));
    if ( out == NULL ){
        exit (-1);
    }

    printf("Call to acc_saxpby ... ");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0;i<N; i++)
    // a*v1 + b*v2 = 1.1 * 1 + 1.41 * (-2) = 1.1 - 2.82 = - 1.72
        saxpby(v1, v2, out, a, b, size);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    
    dt = elapsed_time(&start, &finish);
    printf(": %f s/iter, total time %f s \n", dt/(double)N, dt);
    res[0] = test_result(out, size, -1.72);
    set_vector(out, size, -1.);

    printf("Call to omp_saxpby ... ");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0;i<N; i++)
    // a*v1 + b*v2 = 1.1 * 1 + 1.41 * (-2) = 1.1 - 2.82 = - 1.72
        omp_saxpby(v1, v2, out, a, b, size);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    
    dt = elapsed_time(&start, &finish);
    printf(": %f s/iter, total time %f s \n", dt/(double)N, dt);
    res[1] = test_result(out, size, -1.72);
    set_vector(out, size, -.0002);

    printf("Call to serial_saxpby ... ");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i=0;i<N; i++)
    // a*v1 + b*v2 = 1.1 * 1 + 1.41 * (-2) = 1.1 - 2.82 = - 1.72
        serial_saxpby(v1, v2, out, a, b, size);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    
    dt = elapsed_time(&start, &finish);
    printf(": %f s/iter, total time %f s \n", dt/(double)N, dt);
    res[2] = test_result(out, size, -1.72);

    printf("The array res %d %d %d\n", (int)res[0], (int)res[1], (int)res[2]);
    return (int) res[0];
}

