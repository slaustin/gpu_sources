#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK(call) {   const cudaError_t error = call; if (error != cudaSuccess) { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); exit(1); }}


__global__ void compute_displacement (int *reso_map,float *projection_mat,float *projection_dis,float *eigenvec,int frame_min,int frame_max,int total_points){

        int k=threadIdx.x + blockDim.x * blockIdx.x;
	int kk,reso_count,bin_num;

	reso_count=0;
	if(k>=frame_min && k<frame_max){
		for(kk=0;kk<total_points;kk++){
			if(reso_map[kk]==1){
				bin_num=k+(frame_max*reso_count);
				projection_dis[k]+=(projection_mat[bin_num]*eigenvec[reso_count]);
            			reso_count+=1;
			}
		}
	}

} // End of Global



int main ()
{
  int devCount,blocks,threads,i,ii,iii,line_num,frame_number,counter_cov,max_frame,min_frame,reso_count,reso_sum,bin_number,frame_dimension,total_points,projection_value;
  float variance_value;
  int *reso;
  float *eigenvector,*projection_matrix,*projection_displacement;
  int *dev_reso;
  float *dev_eigenvector,*dev_projection_matrix,*dev_projection_displacement;
  char buf[256];
  FILE* file0=fopen("map_density.dat","r");
  FILE* file=fopen("reso_map.dat","r");
  FILE* file2=fopen("eigenvector.dat","r");
  FILE* file3=fopen("hb_count_matrix.dat","r");
  FILE *ofp;
  char outputFilename[] = "displacement.dat";

CHECK (cudaSetDevice ( 0 ) );

reso_sum=0;
//Read Resolution Map
while (fgets(buf, sizeof (buf), file0)) {
        sscanf (buf, "%i\t%i\t%i",&i,&ii,&iii);
        reso_sum+=1;
}
fclose(file0);
printf("Total Points=%i\n",reso_sum);

//Allocate Resolution Map
reso=(int *)malloc(reso_sum*sizeof(int));
if(reso == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
memset(reso,0,reso_sum*sizeof(int));

total_points=reso_sum;

reso_sum=0;
//Read Resolution Map
while (fgets(buf, sizeof (buf), file)) {
        sscanf (buf, "%i\t%i\t%i\t%i",&i,&ii,&iii,&line_num);
        reso_sum+=1;
}
rewind(file);
printf("Resolution Points=%i\n",reso_sum);

//Read Resolution Map
while (fgets(buf, sizeof (buf), file)) {
        sscanf (buf, "%i\t%i\t%i\t%i",&i,&ii,&iii,&line_num);
	reso[line_num]=1;
}
fclose (file);


//Allocate Eigenvector Array
eigenvector=(float *)malloc(reso_sum*sizeof(float));
if(eigenvector == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
memset(eigenvector,0,reso_sum*sizeof(float));

counter_cov=0;

printf("Reading Input...\n");
//Fill Eigenvector Array
while (fgets(buf, sizeof (buf), file2)) {
        sscanf (buf, "%f",&variance_value);
	eigenvector[counter_cov]=variance_value;
	counter_cov+=1;
}
fclose (file2);

//Determine Max Frame
counter_cov=0;
reso_count=0;
min_frame=99999999;
counter_cov=0;
max_frame=-1;
frame_dimension=0;
while (fgets(buf, sizeof (buf), file3)) {
        sscanf (buf, "%i\t%i",&frame_number,&projection_value);
	max_frame=frame_number;}
rewind(file3);

frame_dimension=max_frame;

//Allocate Projection Matrix
projection_matrix=(float *)malloc(frame_dimension*reso_sum*sizeof(float));
if(projection_matrix == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
memset(projection_matrix,0,frame_dimension*reso_sum*sizeof(float));

//Fill Projection Matrix
max_frame=-1;
min_frame=99999999;
while (fgets(buf, sizeof (buf), file3)) {
        sscanf (buf, "%i\t%i",&frame_number,&projection_value);
	if(max_frame!=frame_number){
		counter_cov=0;
		reso_count=0;}

	if(reso[counter_cov]==1){
		bin_number=(frame_number)+(frame_dimension*reso_count);
		projection_matrix[bin_number]=float(projection_value);
		reso_count+=1;}

        if(frame_number < min_frame){
		min_frame=frame_number;}

	max_frame=frame_number;
	counter_cov+=1;
}
fclose (file3);

counter_cov=counter_cov-1;

//Allocate Projection Displacement Array
projection_displacement=(float *)malloc(frame_dimension*sizeof(float));
if(projection_displacement == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
memset(projection_displacement,0,frame_dimension*sizeof(float));


//Prepare Device Parameters
cudaGetDeviceCount(&devCount);
for (int i = 0; i < devCount; ++i){
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, i);
	threads=devProp.maxThreadsPerBlock;
}

blocks=ceil(float(frame_dimension)/float(threads))+1;
printf("Threads=%i\n",threads);
printf("Blocks=%i\n",blocks);

//Allocate Device Arrays
CHECK (cudaMalloc((void **) &dev_reso, total_points*sizeof(int)) );
CHECK (cudaMalloc((void **) &dev_projection_matrix, frame_dimension*reso_sum*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_projection_displacement, frame_dimension*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_eigenvector, reso_sum*sizeof(float)) );

CHECK (cudaMemcpy(dev_reso, reso, total_points*sizeof(int), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_projection_matrix, projection_matrix, frame_dimension*reso_sum*sizeof(float), cudaMemcpyHostToDevice) );

CHECK (cudaMemcpy(dev_eigenvector,eigenvector, reso_sum*sizeof(float), cudaMemcpyHostToDevice) );

CHECK (cudaMemcpy(dev_projection_displacement, projection_displacement, frame_dimension*sizeof(float), cudaMemcpyHostToDevice) );

compute_displacement<<<blocks,threads>>>(dev_reso,dev_projection_matrix,dev_projection_displacement,dev_eigenvector,min_frame,frame_dimension,total_points);

CHECK (cudaMemcpy(projection_displacement, dev_projection_displacement, (frame_dimension)*sizeof(float), cudaMemcpyDeviceToHost) );

CHECK (cudaFree(dev_reso) );
CHECK (cudaFree(dev_projection_matrix) );
CHECK (cudaFree(dev_projection_displacement) );
CHECK (cudaFree(dev_eigenvector) );
cudaDeviceReset();

//Write a File
ofp=fopen(outputFilename, "w");
for (i=min_frame;i<frame_dimension;i++){
        fprintf(ofp,"%i\t%f\n",i,projection_displacement[i]);
}
fclose(ofp);

//Free Allocated Arrays
free(reso);
free(projection_matrix);
free(projection_displacement);
free(eigenvector);


printf("Complete!\n");
  return 0;
}
