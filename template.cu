#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK(call) {   const cudaError_t error = call; if (error != cudaSuccess) { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); exit(1); }}


__global__ void default_name (float *default_float,int default_int) {

        int k=threadIdx.x + blockDim.x * blockIdx.x;

} // End of Global



int main ()
{
  int devCount,blocks,threads;
  float ;
  int *int_arrays;
  float *float_arrays;
  int *dev_int_arrays;
  float *dev_float_arrays;
  char buf[256];
  FILE* file=fopen("default.dat","r");
  FILE *ofp;
  char outputFilename[] = "default.out";

CHECK (cudaSetDevice ( 0 ) );

//Read a File
while (fgets(buf, sizeof (buf), file)) {
        sscanf (buf, "%i\t%i\t%i",&x,&y,&z);
}
fclose (file);

//Allocate Local Array
default_array=(int *)malloc(SOMESIZE*sizeof(int));
if(default_array == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
memset(default_array,0,SOMESIZE*sizeof(int));

//Write a File
ofp=fopen(outputFilename, "w");
for (k=0;k<points;k++){
	fprintf(ofp,"%f\n",top_sum[k]/bottom_sum[k]);
}
fclose(ofp);


cudaGetDeviceCount(&devCount);
//printf("CUDA Device Query...\n");
//printf("There are %d CUDA devices.\n", devCount);

// Iterate through devices
for (int i = 0; i < devCount; ++i){
	// Get device properties
	//printf("CUDA Device #%d\n", i);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, i);
	//printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
	threads=devProp.maxThreadsPerBlock;
}


blocks=ceil(float(SOMENUMBER)/float(threads))+1;

printf("Threads=%i\n",threads);
printf("Blocks=%i\n",blocks);

//Allocate on Device and Launch and Copy Back
CHECK (cudaMalloc((void **) &dev_covariance, (points*points)*sizeof(float)) );
CHECK (cudaMemcpy(dev_covariance, covariance, (points*points)*sizeof(float), cudaMemcpyHostToDevice) );
compute_covariance<<<blocks,threads>>>(dev_variance,dev_covariance,points);
CHECK (cudaMemcpy(covariance, dev_covariance, (points*points)*sizeof(float), cudaMemcpyDeviceToHost) );
CHECK (cudaFree(dev_covariance) );
CHECK (cudaFree(dev_variance) );
cudaDeviceReset();

//Free Allocated Arrays
free(reso);
free(top_sum);
printf("Complete!\n");
  return 0;
}
