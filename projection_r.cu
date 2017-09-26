#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK(call) {   const cudaError_t error = call; if (error != cudaSuccess) { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); exit(1); }}


__global__ void compute_displacement (float *projection_mat,float *projection_dis,float *eigenvec,int frame_min,int frame_max,int points){

        int k=threadIdx.x + blockDim.x * blockIdx.x;
	int kk,bin_num;

	if(k>=frame_min && k<frame_max){
		for(kk=0;kk<points;kk++){
			bin_num=k+(frame_max*kk);
			projection_dis[k]+=(projection_mat[bin_num]*eigenvec[kk]);
		}
	}

} // End of Global



int main ()
{
  int devCount,blocks,threads,i,frame_number,counter_cov,max_frame,min_frame,reso_count,bin_number,frame_dimension,points,projection_value,frame,atom_index,curr_frame;
  float variance_value,position,bias;
  float *eigenvector,*projection_matrix,*projection_displacement;
  float *dev_eigenvector,*dev_projection_matrix,*dev_projection_displacement;
  char buf[256];
  FILE* file=fopen("selection_coords.dat","r");
  FILE* file2=fopen("eigenvector.dat","r");
  FILE* file3=fopen("atomic_count_matrix.dat","r");
  FILE *ofp;
  char outputFilename[] = "displacement.dat";

CHECK (cudaSetDevice ( 0 ) );


printf("Initilizing...\n");
points=0;
max_frame=0;
while (fgets(buf, sizeof (buf), file)) {
        sscanf (buf, "%i\t%i\t%f\t%f",&frame,&atom_index,&position,&bias);
        if(points==0){curr_frame=frame;}
        if(curr_frame==frame){points+=1;}
        max_frame=frame;
}
fclose(file);

printf("Number of Atoms=%i\n",points/3);
printf("Max Frame=%i\n",max_frame);
printf("Points=%i\n",points);
printf("Points*max_frame=%i\n",points*max_frame);

////////////////////////////////


//Allocate Eigenvector Array
eigenvector=(float *)malloc(points*sizeof(float));
if(eigenvector == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
memset(eigenvector,0,points*sizeof(float));

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

frame_dimension=max_frame;

//Allocate Projection Matrix
projection_matrix=(float *)malloc(frame_dimension*points*sizeof(float));
if(projection_matrix == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}
memset(projection_matrix,0,frame_dimension*points*sizeof(float));

printf("Fill Matrix...\n");

//Fill Projection Matrix
max_frame=-1;
min_frame=99999999;
while (fgets(buf, sizeof (buf), file3)) {
        sscanf (buf, "%i\t%i",&frame_number,&projection_value);
	if(max_frame!=frame_number){
		counter_cov=0;
		reso_count=0;}
		bin_number=(frame_number)+(frame_dimension*reso_count);
		projection_matrix[bin_number]=float(projection_value);
		reso_count+=1;

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
CHECK (cudaMalloc((void **) &dev_projection_matrix, frame_dimension*points*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_projection_displacement, frame_dimension*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_eigenvector, points*sizeof(float)) );

CHECK (cudaMemcpy(dev_projection_matrix, projection_matrix, frame_dimension*points*sizeof(float), cudaMemcpyHostToDevice) );

CHECK (cudaMemcpy(dev_eigenvector,eigenvector, points*sizeof(float), cudaMemcpyHostToDevice) );

CHECK (cudaMemcpy(dev_projection_displacement, projection_displacement, frame_dimension*sizeof(float), cudaMemcpyHostToDevice) );

compute_displacement<<<blocks,threads>>>(dev_projection_matrix,dev_projection_displacement,dev_eigenvector,min_frame,frame_dimension,points);

CHECK (cudaMemcpy(projection_displacement, dev_projection_displacement, (frame_dimension)*sizeof(float), cudaMemcpyDeviceToHost) );

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
free(projection_matrix);
free(projection_displacement);
free(eigenvector);


printf("Complete!\n");
  return 0;
}
