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
  int devCount,blocks,threads,x,y,min_index,max_index;
  float covariance_value;
  unsigned long long int sqpoints,grid_num;
  int *int_arrays;
  float *histogram,histogram_2,histogram_T,result,result_2;
  int *dev_int_arrays;
  float *dev_float_arrays;
  char buf[256];
  FILE* file=fopen("hb_covarinace_matrix.dat","r");
  FILE* file2=fopen("eigenvectors.out","r");
  FILE *ofp;
  char outputFilename[] = "default.out";

CHECK (cudaSetDevice ( 0 ) );


min_index=9999;
max_index=-9999;

//Read a File
while (fgets(buf, sizeof (buf), file)) {
        sscanf (buf, "%i\t%i\t%f",&x,&y,&covariance_value);
	if(x<min_index){min_index=x;}
	if(x>max_index){max_index=x;}
}
rewind (file);

sqpoints=(unsigned long long)max_index*max_index;

//Allocate Local Array
histogram=(int *)malloc(sqpoints*sizeof(float));
if(histogram == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}


histogram_2=(int *)malloc(sqpoints*sizeof(float));
if(histogram_2 == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

histogram_T=(int *)malloc(sqpoints*sizeof(float));
if(histogram_T == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

result=(int *)malloc(sqpoints*sizeof(float));
if(result == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

result_2=(int *)malloc(sqpoints*sizeof(float));
if(result_2 == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

memset(histogram,0,sqpoints*sizeof(float));
memset(histogram_2,0,sqpoints*sizeof(float));
memset(histogram_T,0,sqpoints*sizeof(float));
memset(result,0,sqpoints*sizeof(float));
memset(result_2,0,sqpoints*sizeof(float));

while (fgets(buf, sizeof (buf), file)) {
        sscanf (buf, "%i\t%i\t%f",&x,&y,&covariance_value);
	grid_num=(unsigned long long)y*max_index;
        grid_num+=x;
	histogram[grid_num]=covariance_value;
}
close (file);


while (fgets(buf, sizeof (buf), file2)) {
        sscanf (buf, "%i\t%i\t%f",&x,&y,&covariance_value);
        grid_num=(unsigned long long)y*max_index;
        grid_num+=x;
        histogram_2[grid_num]=covariance_value;
	grid_num=(unsigned long long)x*max_index;
        grid_num+=y;
        histogram_T[grid_num]=covariance_value;
}
close (file2);

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
