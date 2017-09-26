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
  int blocks,threads,line_num,frame,count,k,j,x,y,z,points,grid_point,min_x,min_y,min_z,max_x,max_y,max_z,max_frame,line_counter,avg_only,print_flg,all_points,test_sum;
  unsigned long long int sqpoints,grid_num;
  int devCount;
  float bias,R,T;
  int *reso;
  float *top_sum,*bottom_sum,*covariance,*covariance_2,*variance;
  float *dev_covariance,*dev_variance;
  char buf[4096];
  FILE* file=fopen("hb_count_matrix.dat","r");
  FILE* file2=fopen("reso_map.dat","r");
  FILE* file3=fopen("map_density.dat","r");
  FILE *ofp;
  FILE *ofp2;
  char outputFilename[] = "weighted_avg.dat";
  char outputFilename2[] = "hb_covariance_matrix.dat";

CHECK (cudaSetDevice ( 0 ) );

avg_only=0;
print_flg=1;
R=0.001986;
T=300.00;

min_x=999;
min_y=999;
min_z=999;

max_x=-999;
max_y=-999;
max_z=-999;

points=0;
while (fgets(buf, sizeof (buf), file3)) {
        sscanf (buf, "%i\t%i\t%i",&x,&y,&z);
        points+=1;}

fclose (file3);

reso=(int *)malloc(points*sizeof(int));
if(reso == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
memset(reso,0,points*sizeof(int));

all_points=points;

points=0;
while (fgets(buf, sizeof (buf), file2)) {
	sscanf (buf, "%i\t%i\t%i\t%i",&x,&y,&z,&line_num);
        if(x<min_x){min_x=x;}
        if(y<min_y){min_y=y;}
        if(z<min_z){min_z=z;}
        if(x>max_x){max_x=x;}
        if(y>max_y){max_y=y;}
        if(z>max_z){max_z=z;}
        reso[line_num]=1;
        points+=1;}

fclose (file2);


sqpoints= (unsigned long long )points*points;

test_sum=0;
for (k=0;k<all_points;k++){
	test_sum+=reso[k];
}

printf("~~~~~~~~Box Information~~~~~~~~\n");
printf("Minx=%i\n",min_x);
printf("Miny=%i\n",min_y);
printf("Minz=%i\n",min_z);
printf("Maxx=%i\n",max_x);
printf("Maxy=%i\n",max_y);
printf("Maxz=%i\n",max_z);

printf("Points=%i\n",points);
printf("Check=%i\n",test_sum);
printf("sqpoints=%llu\n",sqpoints);
printf("Check2=%llu\n",sqpoints/points);

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

grid_point=0;
max_frame=0;
line_counter=0;
while (fgets(buf, sizeof (buf), file)) {
	if(line_counter==all_points){
		grid_point=0;
		line_counter=0;}
	sscanf (buf, "%i\t%i\t%f",&frame,&count,&bias);
        if(frame>max_frame){max_frame=frame;}
        if(reso[line_counter]==1){
        	top_sum[grid_point]+=(expf((-1.0*bias)/(R*T))*float(count));
        	bottom_sum[grid_point]+=(expf(((-1.0*bias)/(R*T))));
        	grid_point+=1;}
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
		CHECK (cudaMemcpy(dev_covariance, covariance, (sqpoints)*sizeof(float), cudaMemcpyHostToDevice) );
		CHECK (cudaMemcpy(dev_variance, variance, points*sizeof(float), cudaMemcpyHostToDevice) );
		compute_covariance<<<blocks,threads>>>(dev_variance,dev_covariance,points,bias);
		CHECK (cudaMemcpy(covariance, dev_covariance, (sqpoints)*sizeof(float), cudaMemcpyDeviceToHost) );
		grid_point=0;
		line_counter=0;}
        if(reso[line_counter]==1){
        	sscanf (buf, "%i\t%i\t%f",&frame,&count,&bias);
        	variance[grid_point]=(float(count)-(top_sum[grid_point]/bottom_sum[grid_point]));
        	grid_point+=1;}
	line_counter+=1;
}


rewind(file);

CHECK (cudaMemcpy(dev_covariance, covariance_2, (sqpoints)*sizeof(float), cudaMemcpyHostToDevice) );

grid_point=0;
line_counter=0;
printf("Compute Covariance_2...\n");
while (fgets(buf, sizeof (buf), file)) {
       if(line_counter==all_points){
                CHECK (cudaMemcpy(dev_covariance, covariance_2, (sqpoints)*sizeof(float), cudaMemcpyHostToDevice) );
                compute_covariance_2<<<blocks,threads>>>(dev_covariance,points,bias);
                CHECK (cudaMemcpy(covariance_2, dev_covariance, (sqpoints)*sizeof(float), cudaMemcpyDeviceToHost) );
                grid_point=0;
                line_counter=0;}
        if(reso[line_counter]==1){
                sscanf (buf, "%i\t%i\t%f",&frame,&count,&bias);
                grid_point+=1;}
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
			grid_num=(unsigned long long) (j*points);
			grid_num+=k;
			fprintf(ofp2,"%i\t%i\t%f\n",k+1,j+1,(covariance[grid_num]/covariance_2[grid_num]));
		}
	}
fclose(ofp2);
}

}//Avg_only

free(reso);
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
