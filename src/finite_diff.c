#include <math.h>
#include <stdlib.h>
#include <stdio.h>

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
        for (i=0;i<nx; i++){
            for (j=0;j<ny-1; j++){
                index_low = i + j * nx;
                index_high = i + (j+1)*nx;
                outimage[index_low] = inimage[index_high] - inimage[index_low];
            }
            // boundary condition: nearest
            outimage[index_high] = outimage[index_low];
        }            
    }

    

}