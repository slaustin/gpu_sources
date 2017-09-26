#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK(call) {   const cudaError_t error = call; if (error != cudaSuccess) { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); exit(1); }}


__global__ void compute_covariance (float *variance,float *covariance,int points,float bias) {

        int k=threadIdx.x + blockDim.x * blockIdx.x;
        int j;
	unsigned long long int grid_num;
	float R,T;

	R=0.00198588;
	T=300.0;

	if(k<points){
		for (j=0;j<points;j++){
			grid_num=(unsigned long long)j*points;
			grid_num+=k;
			covariance[grid_num]+=(variance[k]*variance[j])*expf((-1.0*bias)/(R*T));
		}
	}
} // End of Global




__global__ void compute_covariance_2 (float *covariance,int points,float bias) {

        int k=threadIdx.x + blockDim.x * blockIdx.x;
        int j;
	unsigned long long int grid_num;
	float R,T;

	R=0.00198588;
	T=300.0;

        if(k<points){
                for (j=0;j<points;j++){
			grid_num=(unsigned long long)j*points;
			grid_num+=k;
                        covariance[grid_num]+=expf((-1.0*bias)/(R*T));
                }
        }
} // End of Global


int main ()
{
  int blocks,threads,frame,k,j,points,grid_point,curr_frame,max_frame,atom_index,line_counter,avg_only,print_flg,all_points;
  unsigned long long int sqpoints,grid_num;
  int devCount;
  float bias,R,T,count,position;
  float *top_sum,*bottom_sum,*covariance,*covariance_2,*variance;
  float *dev_covariance,*dev_variance;
  char buf[4096];
  FILE* file=fopen("selection_coords.dat","r");
  FILE *ofp;
  FILE *ofp2;
  char outputFilename[] = "weighted_avg_position.dat";
  char outputFilename2[] = "atomic_covariance_matrix.dat";

CHECK (cudaSetDevice ( 0 ) );

avg_only=0;
print_flg=1;
R=0.001986;
T=300.00;

printf("Initilizing...\n");
points=0;
max_frame=0;
while (fgets(buf, sizeof (buf), file)) {
        sscanf (buf, "%i\t%i\t%f\t%f",&frame,&atom_index,&position,&bias);
	if(points==0){curr_frame=frame;}
	if(curr_frame==frame){points+=1;}
	max_frame=frame;
}

all_points=points;
printf("Number of Atoms=%i\n",points/3);
printf("Max Frame=%i\n",max_frame);

sqpoints= (unsigned long long )points*points;

top_sum=(float *)malloc(points*sizeof(float));
if(top_sum == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

bottom_sum=(float *)malloc(points*sizeof(float));
if(bottom_sum == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}


if(avg_only == 0){
variance=(float *)malloc(points*sizeof(float));
if(variance == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

covariance=(float *)malloc(sqpoints*sizeof(float));
if(covariance == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

covariance_2=(float *)malloc(sqpoints*sizeof(float));
if(covariance_2 == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
}


printf("Set Memory...\n");

memset(top_sum,0,points*sizeof(float));
memset(bottom_sum,0,points*sizeof(float));
if(avg_only == 0){
memset(variance,0,points*sizeof(float));
memset(covariance,0,(sqpoints)*sizeof(float));
memset(covariance_2,0,(sqpoints)*sizeof(float));
}
printf("Reading Input...\n");


rewind(file);
grid_point=0;
line_counter=0;
while (fgets(buf, sizeof (buf), file)) {
	if(line_counter==all_points){
		grid_point=0;
		line_counter=0;}
	sscanf (buf, "%i\t%i\t%f\t%f",&frame,&atom_index,&position,&bias);
        top_sum[grid_point]+=(expf((-1.0*bias)/(R*T))*float(position));
        bottom_sum[grid_point]+=(expf(((-1.0*bias)/(R*T))));
        grid_point+=1;
        line_counter+=1;
}

printf("Write Average...\n");
ofp=fopen(outputFilename, "w");
for (k=0;k<points;k++){
	fprintf(ofp,"%f\n",top_sum[k]/bottom_sum[k]);
}
fclose(ofp);

//Avg Only Below
if(avg_only == 0){

cudaGetDeviceCount(&devCount);

// Iterate through devices
for (int i = 0; i < devCount; ++i){
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, i);
	threads=devProp.maxThreadsPerBlock;
}


blocks=ceil(float(points)/float(threads))+1;

printf("Threads=%i\n",threads);
printf("Blocks=%i\n",blocks);

CHECK (cudaMalloc((void **) &dev_covariance, (sqpoints)*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_variance, points*sizeof(float)) );
CHECK (cudaMemcpy(dev_covariance, covariance, (sqpoints)*sizeof(float), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_variance, variance, points*sizeof(float), cudaMemcpyHostToDevice) );


rewind(file);
grid_point=0;
line_counter=0;
printf("Compute Covariance...\n");
while (fgets(buf, sizeof (buf), file)) {
       if(line_counter==all_points){
		if(frame%100==0){printf("Frame=%i\n",frame);}
		CHECK (cudaMemcpy(dev_covariance, covariance, (sqpoints)*sizeof(float), cudaMemcpyHostToDevice) );
		CHECK (cudaMemcpy(dev_variance, variance, points*sizeof(float), cudaMemcpyHostToDevice) );
		compute_covariance<<<blocks,threads>>>(dev_variance,dev_covariance,points,bias);
		CHECK (cudaMemcpy(covariance, dev_covariance, (sqpoints)*sizeof(float), cudaMemcpyDeviceToHost) );
		grid_point=0;
		line_counter=0;}
	sscanf (buf, "%i\t%i\t%f\t%f",&frame,&atom_index,&position,&bias);
        variance[grid_point]=(float(position)-(top_sum[grid_point]/bottom_sum[grid_point]));
	grid_point+=1;
	line_counter+=1;
}

CHECK (cudaMemcpy(dev_covariance, covariance_2, (sqpoints)*sizeof(float), cudaMemcpyHostToDevice) );

rewind(file);
grid_point=0;
line_counter=0;
printf("Compute Covariance_2...\n");
while (fgets(buf, sizeof (buf), file)) {
       if(line_counter==all_points){
		if(frame%100==0){printf("Frame=%i\n",frame);}
                CHECK (cudaMemcpy(dev_covariance, covariance_2, (sqpoints)*sizeof(float), cudaMemcpyHostToDevice) );
                compute_covariance_2<<<blocks,threads>>>(dev_covariance,points,bias);
                CHECK (cudaMemcpy(covariance_2, dev_covariance, (sqpoints)*sizeof(float), cudaMemcpyDeviceToHost) );
                grid_point=0;
                line_counter=0;}
	sscanf (buf, "%i\t%i\t%f\t%f",&frame,&atom_index,&position,&bias);
        variance[grid_point]=(float(position)-(top_sum[grid_point]/bottom_sum[grid_point]));
        grid_point+=1;
        line_counter+=1;
}

fclose (file);


CHECK (cudaFree(dev_covariance) );
CHECK (cudaFree(dev_variance) );
cudaDeviceReset();


if(print_flg==1){
printf("Write Covariance...\n");
ofp2=fopen(outputFilename2, "w");
for (k=0;k<points;k++){
	for (j=0;j<points;j++){
		grid_num=(unsigned long long)j*points;
		grid_num+=k;
		fprintf(ofp2,"%i\t%i\t%f\n",k+1,j+1,(covariance[grid_num]/covariance_2[grid_num]));
	}
}
fclose(ofp2);
}
}//Avg_only

free(top_sum);
free(bottom_sum);
if(avg_only == 0){
free(covariance);
free(covariance_2);
free(variance);
}
printf("Complete!\n");
  return 0;
}
