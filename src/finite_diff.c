#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "omp.h"

int main(int argc, char **argv){

   printf("Hello World!\n" );
   return 0;

}

int fdiff_for(float *inimage, float *outimage, long nx, long ny, int direction ){
    long index_low, index_high, i, j;
    if (direction == 0){
        // assuming the image is stored as C array Y,X
        for (j=0;j<ny; j++){
            for (i=0;i<nx-1; i++){
                index_low = i + j * nx;
                index_high = index_low + 1;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            }
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }
        
    } else if (direction == 1) {
        for (j=0;j<ny-1; j++){
            for (i=0;i<nx; i++){
            
                index_low = i + j * nx;
                index_high = i + (j+1)*nx;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            }
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }            
    }

    return 0;

}

int fdiff_for_unrolled(float *inimage, float *outimage, long nx, long ny, int direction ){
    long index_low, index_high, i, j;
    if (direction == 0){
        // assuming the image is stored as C array Y,X
        for (i=0;i<ny*(nx-1); i++){
            //for (i=0;i<nx-1; i++){
                index_low = i ;
                index_high = index_low + 1;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            //}
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }
        
    } else if (direction == 1) {
        for (i=0;i<nx*(ny-1); i++){
            //for (j=0;j<ny-1; j++){
                index_low = i ;
                index_high = index_low + nx;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            //}
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }            
    }

    return 0;

}

int fdiff_for_simd(float *inimage, float *outimage, long nx, long ny, int direction ){
    long index_low, index_high, i, j;
    if (direction == 0){
        // assuming the image is stored as C array Y,X
        #pragma omp simd
        for (i=0;i<ny*(nx-1); i++){
            //for (i=0;i<nx-1; i++){
                index_low = i ;
                index_high = index_low + 1;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            //}
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }
        
    } else if (direction == 1) {
        #pragma omp simd
        for (i=0;i<nx*(ny-1); i++){
            //for (j=0;j<ny-1; j++){
                index_low = i ;
                index_high = index_low + nx;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            //}
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }            
    }

    return 0;

}

int fdiff_for_parallel(float *inimage, float *outimage, long nx, long ny, int direction ){
    long index_low, index_high, i,j;
    if (direction == 0){
        // assuming the image is stored as C array Y,X
        #pragma omp parallel for private(index_high, index_low) shared(i)
        for (i=0;i<ny*(nx-1); i++){
            //for (i=0;i<nx-1; i++){
                index_low = i ;
                index_high = index_low + 1;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            //}
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }
        
    } else if (direction == 1) {
        #pragma omp parallel for private(index_high, index_low) shared(i,j)
        for (j=0;j<ny-1; j++){
            for (i=0;i<nx; i++){
                index_low = i + j * nx;
                index_high = i + (j+1)*nx;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            }
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }            
    }

    return 0;

}

int fdiff_parallel_for_simd(float *inimage, float *outimagex, float *outimagey, long nx, long ny){
    int i=0;
    float *outimages[2];
    outimages[0] = outimagex;
    outimages[1] = outimagey;
    int ret[2];
    #pragma omp parallel for
    for (i=0;i<2;i++)
    {
        ret[i] = fdiff_for_simd(inimage, outimages[i], nx, ny, i);
        //ret[1] = fdiff_for_unrolled(inimage, outimagey, nx, ny, 1);
    }
    return ret[0]+ret[1];
}

int fdiff_parallel_whole(float *inimage, float *outimagex, float *outimagey, long nx, long ny){
    int i=0;
    float *outimages[2];
    outimages[0] = outimagex;
    outimages[1] = outimagey;
    int ret[2];
    #pragma omp parallel for
    for (i=0;i<2;i++)
    {
        ret[i] = fdiff_for(inimage, outimages[i], nx, ny, i);
        //ret[1] = fdiff_for_unrolled(inimage, outimagey, nx, ny, 1);
    }
    return ret[0]+ret[1];
}


int fdiff_for_parallel_concurrent(float *inimage, float *outimagex, float *outimagey,  long nx, long ny){
    long index_low, index_high_x, index_high_y, i,j;
    
    #pragma omp parallel for private(index_high_x, index_high_y, index_low) shared(i,j)
    for (j=0;j<ny-1; j++){
        for (i=0;i<nx; i++){
            index_low = i + j * nx;
            index_high_x = index_low+1;
            index_high_y = index_low + nx;
            outimagex[index_low] = inimage[index_high_x] - inimage[index_low];
            outimagey[index_low] = inimage[index_high_y] - inimage[index_low];
        }
        // boundary condition: nearest
        outimagex[index_high_x] = outimagex[index_low];
        outimagey[index_high_y] = outimagey[index_low];
    }            

    return 0;

}
