/* 
   Sample Solution to the game of life program using MPI.
   Author: Purushotham Bangalore
   Date: Feb 17, 2010

   Use -DDEBUG0 for printing local size, prev/next rank, counts, displs.
   Use -DDEBUG1 for output at the start and end.
   Use -DDEBUG2 for output at each iteration.

   To compile: mpicc -Wall -O -o mpi_life mpi_life_1d.c
   nvcc -O3  -I/opt/asn/apps/openmpi_4.1.4_gcc11_cuda/include -DDEBUG0 -o cuda_aware_MPI_life cuda_aware_MPI_life.cu -lmpi
     
   To run: 
   Local system: 
           mpirun -np <#P> ./mpi_life <problem size> <max iters> <output dir>
   In my case: mpirun -np 2 ./mpi_life 10000 5000 /scratch/ualclsb0056
   -- To store the file in the scratch, you need to create a new folder (use your username), use this command: 
   mkdir /scratch/ualclsb00##
   To check the output file, use this command:
   cd  /scratch/ualclsb00##
   To load the module, you can load openmpi without CUDA 
   DMC Cluster at ASC:
           Follow instructions to execute MPI programs on DMC cluster
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define DIES   0
#define ALIVE  1
const int TILE_DIM = 32;

/* function to measure time taken */
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

/* allocate row-major two-dimensional array */
int *allocarray(int P, int Q) {
 
  int *p;

  p = (int *)malloc(P*Q*sizeof(int));
//   a = (int **)malloc(P*sizeof(int*));
//   for (i = 0; i < P; i++)
//     a[i] = &p[i*Q]; 

  return p;
}

/* free allocated memory */
void freearray(int *a) {
  //free(&a[0][0]);
  free(a);
}

/* print arrays in 2D format */
void printarray(int *a, int M, int N, int k) {
  int i, j;
  fprintf(stderr,"Life after %d iterations:\n", k) ;
  for (i = 0; i < M+2; i++) {
    for (j = 0; j< N+2; j++)
      fprintf(stderr,"%d ", a[i*(N+2)+j]);
    fprintf(stderr,"\n");
  }
  fprintf(stderr,"\n");
}

/* write array to a file (including ghost cells) */
void writefile(int *a, int N, FILE *fptr) {
  int i, j;
  for (i = 0; i < N+2; i++) {
    for (j = 0; j< N+2; j++)
      fprintf(fptr, "%d ", a[i*(N+2)+j]);
    fprintf(fptr, "\n");
  }
}

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
__global__
void compute_gpu(int *life, int *temp, int M, int N) {
  

  int x = blockIdx.x * blockDim.x + threadIdx.x+1;
  int y = blockIdx.y * blockDim.y + threadIdx.y+1;


  //int width = gridDim.x * TILE_DIM;   
  int id= x*(N+2)+y;
  int neighbors;
  //printf("gridDim=%d width=%d\n",gridDim,width);
  if(x<=M &&y<=N){
      neighbors = life[id + (N + 2)] +                           // Upper neighbor
                    life[id - (N + 2)] +                           // Lower neighbor
                    life[id + 1] +                                      // Right neighbor
                    life[id - 1] +                                      // Left neighbor
                    life[id + (N + 3)] + life[id - (N + 3)] + // Diagonal neighbors
                    life[id - (N + 1)] + life[id + (N + 1)];
        
        // if(x==5 && y==7)
        //   printf("value of life[%d][%d]=%d\n",x,y,neighbors);
        temp[id] = (neighbors == 3 || (neighbors == 2 && life[id]))? 1 : 0;
        //printf("neighbors=%d x=%d y=%d id=%d life=%d temp=%d\n",neighbors,x,y,id, life[id],temp[id]);
    
  }
  
   }
/* update each cell based on old values */
int compute(int *life, int *temp, int M, int N) {
  int i, j, value, flag=0;

  for (i = 1; i < M+1; i++) {
    for (j = 1; j < N+1; j++) {
      /* find out the value of the current cell */
      value = life[(i-1) * (N+2)+(j-1)] + life[(i-1) * (N+2)+j] + life[(i-1) * (N+2)+(j+1)]
            + life[i * (N+2)+(j-1)]                   + life[i * (N+2)+(j+1)]
              + life[(i+1) * (N+2)+(j-1)] + life[(i+1) * (N+2)+j] + life[(i+1) * (N+2)+(j+1)] ;
                  
      
      /* check if the cell dies or life is born */
      if (life[i*(N+2)+j]) { // cell was alive in the earlier iteration
        if (value < 2 || value > 3) {
          temp[i*(N+2)+j] = DIES ;
          flag++; // value changed 
        }
        else // value must be 2 or 3, so no need to check explicitly
          temp[i * (N+2)+j] = ALIVE ; // no change
            } 
      else { // cell was dead in the earlier iteration
        if (value == 3) {
          temp[i * (N+2)+j] = ALIVE;
          flag++; // value changed 
        }
        else
          temp[i * (N+2)+j] = DIES; // no change
      }
    }
  }

  return flag;
}

/* function to exchange boundary rows */
// void exchange(int *mylife, int myN, int N, int prev, int next) {
//   // implement this function
  
//   MPI_Recv(&mylife[(myN+1)*(N+2)+0], N+2, MPI_INT, next, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//   MPI_Send(&mylife[1*(N+2)+0], N+2, MPI_INT, prev, 0, MPI_COMM_WORLD);
  
  
  
//   MPI_Recv(&mylife[0*(N+2)+0], N+2, MPI_INT, prev, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//   MPI_Send(&mylife[myN*(N+2)+0], N+2, MPI_INT, next, 0, MPI_COMM_WORLD);
  
  
// }


int main(int argc, char **argv) {
  int N, NTIMES, *life=NULL, *temp=NULL;
  //int *ptr ;
  int *d_life=NULL,*d_temp=NULL,*h_life=NULL,*h_temp=NULL;
  int i, j, k, rank, size, prev, next;
  //int flag=1, myflag;
  int myN, *mylife=NULL, *bufptr=NULL, *counts=NULL, *displs=NULL;
  double t1, t2;
  char filename[BUFSIZ];
  FILE *fptr=NULL;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 4) {
    printf("Usage: %s <problem size> <max. iterations> <output dir>\n", 
	   argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  N = atoi(argv[1]);
  NTIMES = atoi(argv[2]);

  /* Compute the rows per process */
  myN = N/size + ((rank < (N%size))?1:0);

  /* Compute ranks of neighboring processes */
  prev = rank - 1;
  next = rank + 1;
  if (rank == 0) prev = MPI_PROC_NULL;
  if (rank == size-1) next = MPI_PROC_NULL;

#ifdef DEBUG0
  printf("[%d]: myN = %d prev = %d next = %d\n", rank, myN, prev, next);
#endif 
  
  /* Allocate memory for local life array and temp array */
  mylife = allocarray(myN+2,N+2);
  temp = allocarray(myN+2,N+2);
  
  /* Initialize the boundaries of the temp matrix */
  for (i = 0; i < myN+2; i++) {
    temp[i*(N+2)+0] = temp[i*(N+2)+N+1] = DIES ;
  }
  for (j = 0; j < N+2; j++) {
    temp[0*(N+2)+j] = temp[(myN+1)*(N+2)+j] = DIES ;
  }
  
  
  for (i = 0; i < N+2; i++) {
       mylife[0*(N+2)+i] = mylife[(myN+1)*(N+2)+i] =DIES ;
  }
  
  
  dim3 dimGrid((N+2+TILE_DIM-1)/TILE_DIM, (N+2+TILE_DIM-1)/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
  // printf("dimGrid: %d %d %d. dimBlock: %d %d %d\n",
  //        dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
  
  if(rank==0){
      /* compute the count and displacement values to scatter the array */
  counts = (int *)malloc(sizeof(int)* size);
  displs = (int *)malloc(sizeof(int)* size);

  for (i = 0; i < size; i++)
    counts[i] = ((N/size) + ((i < (N%size))?1:0))*(N+2);
  displs[0] = 0;
  for (i = 1; i < size; i++)
    displs[i] = displs[i-1] + counts[i-1];
    
  }
  
  if (rank == 0) {
  
  
  
  
  
    /* Allocate memory for full life array */
    life = allocarray(N+2,N+2);

    /* Initialize the boundaries of the life array */
    for (i = 0; i < N+2; i++) {
      life[0*(N+2)+i] = life[i*(N+2)+0] = life[(N+1)*(N+2)+i] = life[i*(N+2)+(N+1)] = DIES ;
      
    }

    /* Initialize the life array */
    for (i = 1; i < N+1; i++) {
      srand48(54321|i);
      for (j = 1; j< N+1; j++)
        if (drand48() < 0.5) 
          life[i*(N+2)+j] = ALIVE ;
        else
          life[i*(N+2)+j] = DIES ;
    }

#ifdef DEBUG1
    /* Display the initialized life matrix */
    printarray(life, N, N, 0);
#endif
#ifdef DEBUG2
    /* Print no. of cells alive after the current iteration */
    fprintf(stderr, "Before\n");
    fprintf(stderr," rank=%d myN=%d line=%d\n",rank,myN,__LINE__) ;
    k=0;
    /* Display the life matrix */
    printarray(life, N, N, k+1);
#endif
#ifdef DEBUG0
    for (i = 0; i < size; i++)
      printf("rank=%d: counts[%d] = %d displs[%d] =  %d\n", 
             i, i, counts[i], i, displs[i]);
#endif 
 /* set starting address of send buffer */
    bufptr = &life[1*(N+2)+0];
    
  } 
  
    



   

    t1 = MPI_Wtime();
  /* distribute the life array using 1-D distribution across rows */
     MPI_Scatterv(bufptr, counts, displs, MPI_INT, 
               &mylife[1*(N+2)+0], myN*(N+2), MPI_INT, 0, MPI_COMM_WORLD);
#ifdef DEBUG2
    /* Print no. of cells alive after the current iteration */
    fprintf(stderr, "after scatterv\n");
    fprintf(stderr," rank=%d myN=%d line=%d\n",rank,myN,__LINE__) ;
    k=0;
    /* Display the life matrix */
    printarray(mylife, myN, N, k+1);
#endif
    cudaMalloc(&d_life, (myN+2)*(N+2)*sizeof(int)); 
    cudaMalloc(&d_temp, (myN+2)*(N+2)*sizeof(int));
    h_life = (int *)malloc((myN+2)*(N+2)*sizeof(int));
    h_temp = (int *)malloc((myN+2)*(N+2)*sizeof(int));
    cudaMemcpy(d_life, mylife, (myN+2)*(N+2)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, temp, (myN+2)*(N+2)*sizeof(int), cudaMemcpyHostToDevice);
   
     cudaEvent_t startEvent, stopEvent;
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  float ms,ms1;
checkCuda( cudaEventRecord(startEvent, 0) );
// t1_gpu = gettime();
  for (k = 0; k < NTIMES; k += 2) {

    //compute_shared_gpu<<<threads, grid, shared_size>>>(d_life, d_temp, width, height);
    //compute_shared_gpu<<<threads, grid, shared_size>>>(d_temp, d_life, width, height);
    //compute_shared_gpu<<<dimGrid,dimBlock>>>(d_life,d_temp,N,N);
    //checkCuda(cudaDeviceSynchronize());
    //compute_shared_gpu<<<dimGrid,dimBlock>>>(d_temp,d_life,N,N);
    cudaMemcpy(h_life, d_life, (myN+2)*(N+2)*sizeof(int), cudaMemcpyDeviceToHost); 
#ifdef DEBUG2
    /* Print no. of cells alive after the current iteration */
    fprintf(stderr, "Before1\n");
    fprintf(stderr,"No. of cells whose value changed in iteration %d rank=%d myN=%d\n",
	   k+1,rank,myN) ;
    
    /* Display the life matrix */
    printarray(h_life, myN, N, k+1);
#endif

    MPI_Recv(&h_life[(myN+1)*(N+2)+0], N+2, MPI_INT, next, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(&h_life[1*(N+2)+0], N+2, MPI_INT, prev, 0, MPI_COMM_WORLD);
    
    MPI_Recv(&h_life[0*(N+2)+0], N+2, MPI_INT, prev, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(&h_life[myN*(N+2)+0], N+2, MPI_INT, next, 0, MPI_COMM_WORLD);

#ifdef DEBUG2
    /* Print no. of cells alive after the current iteration */
    fprintf(stderr, "After\n");
    fprintf(stderr,"No. of cells whose value changed in iteration %d rank=%d myN=%d\n",
	   k+1,rank,myN) ;
    
    /* Display the life matrix */
    printarray(h_life, myN, N, k+1);
#endif
    
    cudaMemcpy(d_life, h_life, (myN+2)*(N+2)*sizeof(int), cudaMemcpyHostToDevice);
    compute_gpu<<<dimGrid,dimBlock>>>(d_life,d_temp,myN,N);

    checkCuda(cudaDeviceSynchronize());
    
    cudaMemcpy(h_temp,d_temp,(myN+2)*(N+2)*sizeof(int),cudaMemcpyDeviceToHost);

    MPI_Recv(&h_temp[(myN+1)*(N+2)+0], N+2, MPI_INT, next, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(&h_temp[1*(N+2)+0], N+2, MPI_INT, prev, 0, MPI_COMM_WORLD);
    
    MPI_Recv(&h_temp[0*(N+2)+0], N+2, MPI_INT, prev, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(&h_temp[myN*(N+2)+0], N+2, MPI_INT, next, 0, MPI_COMM_WORLD);
    
    cudaMemcpy(d_temp, h_temp, (myN+2)*(N+2)*sizeof(int), cudaMemcpyHostToDevice);

    //checkCuda(cudaDeviceSynchronize());
    compute_gpu<<<dimGrid,dimBlock>>>(d_temp,d_life,myN,N);
    //cudaMemcpy(h_life,d_life,myN*(N+2)*sizeof(int),cudaMemcpyDeviceToHost);

    checkCuda(cudaDeviceSynchronize());
    //  compute_gpu<<<numBlocks,blockSize>>>(d_life,d_temp,N,N);
    //  compute_gpu<<<numBlocks,blockSize>>>(d_temp,d_life,N,N);



  }

checkCuda( cudaEventRecord(stopEvent, 0) );
checkCuda( cudaEventSynchronize(stopEvent) );
checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );

MPI_Reduce(&ms, &ms1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if(rank==0)
printf("GPU time is taken=%f ms\n",ms1);

cudaMemcpy(h_life, d_life, (myN+2)*(N+2)*sizeof(int), cudaMemcpyDeviceToHost); 
cudaMemcpy(h_temp, d_temp, (myN+2)*(N+2)*sizeof(int), cudaMemcpyDeviceToHost);

  /* Play the game of life for given number of iterations */
//   for (k = 0; k < NTIMES && flag != 0; k++) {
//   for (k = 0; k < NTIMES ; k++) {
//     #ifdef DEBUG0
//         if (rank == 0) printf("[%d]: starting iteration %d\n", rank, k);
//     #endif 
//     /* exchange boundary values */
//     exchange(mylife, myN, N, prev, next);
     
//     /* compute local array */
//     myflag = compute(mylife, temp, myN, N);

//     /* compute global flag value */
//    // MPI_Allreduce(&myflag, &flag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

//     /* copy the new values to the old array */
//     ptr = mylife;
//     mylife = temp;
//     temp = ptr;
  

//   }
 
  /* collect the local life array back into life array */
  MPI_Gatherv(&h_life[1*(N+2)+0], myN*(N+2), MPI_INT, 
	      bufptr, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

  t2 = MPI_Wtime() - t1;
  MPI_Reduce(&t2, &t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


#ifdef DEBUG1
  /* Display the life matrix after k iterations */
  if(rank==0){
  printarray(life, N, N, k);
  }
#endif

  if (rank == 0) {
    //printf("Time taken %f seconds for %d iterations\n", t1, k);
    
    /* open file to write output */
    sprintf(filename,"%s/cuda_MPI_output.%d.%d.%d", argv[3], N, NTIMES, size);
    if ((fptr = fopen(filename, "w")) == NULL) {
      printf("Error opening file %s for writing\n", filename);
      perror("fopen");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Write the final array to output file */
    printf("Writing output to file: %s\n", filename);
    writefile(life, N, fptr);
    fclose(fptr);
    freearray(life);
  }
  cudaFree(d_life);
  cudaFree(d_temp);
  //free(h_life); 
  free(h_life);
  free(h_temp);
  freearray(mylife);
  freearray(temp);

  MPI_Finalize();

  return 0;
}


